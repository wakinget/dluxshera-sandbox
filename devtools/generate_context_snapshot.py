from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import io
import json
from pathlib import Path
from typing import Any, Dict, Optional

from devtools.introspection import print_tree, build_project_index


def _get_repo_root(explicit_root: Optional[str] = None) -> Path:
    """
    Resolve the repository root.

    If `explicit_root` is provided, it is interpreted relative to the
    current working directory (or as an absolute path). Otherwise we
    assume the repo root is one level above this file's directory:

        repo_root/
          ├─ devtools/
          │   ├─ generate_context_snapshot.py
          │   └─ ...
          ├─ src/
          └─ ...

    """
    if explicit_root is not None:
        return Path(explicit_root).resolve()
    return Path(__file__).resolve().parents[1]


def _default_snapshot_dir(repo_root: Path) -> Path:
    """
    Construct a default snapshot directory path under `devtools/`.

    Example:
        devtools/context_snapshot_20251211-141530/
    """
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return repo_root / "devtools" / f"context_snapshot_{ts}"


def _ensure_dir(path: Path) -> None:
    """
    Create a directory (and parents) if needed.

    Raises FileExistsError if the path already exists and is not empty,
    to avoid silently overwriting previous snapshots.
    """
    path.mkdir(parents=True, exist_ok=True)
    # Optionally, you could add a stronger check here if you want to
    # forbid re-using non-empty directories.


def _write_project_tree(
    repo_root: Path,
    out_path: Path,
    max_depth: int,
) -> None:
    """
    Capture the ASCII project tree into `out_path` using print_tree().
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_tree(repo_root, max_depth=max_depth)
    out_path.write_text(buf.getvalue(), encoding="utf-8")


def _write_project_index(
    repo_root: Path,
    out_path: Path,
    max_depth: int,
) -> None:
    """
    Build the project index and write it to `out_path` as JSON.

    This mirrors the behaviour of save_project_index_json, but we go
    through build_project_index directly so we can also embed a subset
    of the index into the snapshot metadata if desired.
    """
    index: Dict[str, Any] = build_project_index(repo_root, max_depth=max_depth)
    out_path.write_text(json.dumps(index, indent=2), encoding="utf-8")


def _collect_basic_metadata(
    repo_root: Path,
    snapshot_dir: Path,
    tree_path: Path,
    index_path: Path,
) -> Dict[str, Any]:
    """
    Assemble a minimal but useful metadata dictionary describing the snapshot.

    This is intentionally lightweight for the first draft. Future versions
    can be extended with ParamSpec/transform/SystemGraph summaries without
    breaking callers.
    """
    now = _dt.datetime.now()

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(repo_root))
        except ValueError:
            return str(p)

    meta: Dict[str, Any] = {
        "schema_version": "0.1",
        "generated_at": now.isoformat(timespec="seconds"),
        "repo_root": str(repo_root),
        "snapshot_dir": rel(snapshot_dir),
        "files": {
            "project_tree": rel(tree_path),
            "project_index": rel(index_path),
            "context_snapshot": rel(snapshot_dir / "context_snapshot.json"),
        },
        "notes": [
            "First-draft context snapshot: includes project tree + static code index.",
            "Future revisions may add ParamSpec summaries, transform registry state, "
            "SystemGraph topology, and test coverage maps.",
        ],
    }

    # Record basic info about the Working Plan, if present.
    working_plan = repo_root / "dLuxShera_Refactor_Working_Plan.md"
    if working_plan.exists():
        stat = working_plan.stat()
        meta["working_plan"] = {
            "exists": True,
            "path": rel(working_plan),
            "size_bytes": stat.st_size,
            "mtime_iso": _dt.datetime.fromtimestamp(stat.st_mtime).isoformat(
                timespec="seconds"
            ),
        }
    else:
        meta["working_plan"] = {"exists": False}

    return meta


def generate_context_snapshot(
    *,
    root: Optional[str] = None,
    out_dir: Optional[str] = None,
    max_depth: int = 4,
) -> Dict[str, Any]:
    """
    Generate a context snapshot for the current dLuxShera repo.

    Parameters
    ----------
    root : str, optional
        Repository root. If omitted, we assume this file lives in
        `<repo_root>/devtools/` and infer the root accordingly.
    out_dir : str, optional
        Snapshot directory. If omitted, a timestamped directory will be
        created under `devtools/`, e.g. `devtools/context_snapshot_YYYYmmDD-HHMMSS`.
        Relative paths are interpreted relative to the repo root.
    max_depth : int, optional
        Maximum directory depth to recurse into when building the project
        tree and index.

    Returns
    -------
    metadata : dict
        A dictionary describing the snapshot contents and paths. This is
        also written to `context_snapshot.json` inside the snapshot dir.
    """
    repo_root = _get_repo_root(root)
    if out_dir is None:
        snapshot_dir = _default_snapshot_dir(repo_root)
    else:
        out_path = Path(out_dir)
        if not out_path.is_absolute():
            snapshot_dir = (repo_root / out_path).resolve()
        else:
            snapshot_dir = out_path

    _ensure_dir(snapshot_dir)

    tree_path = snapshot_dir / "project_tree.txt"
    index_path = snapshot_dir / "project_index.json"
    meta_path = snapshot_dir / "context_snapshot.json"

    _write_project_tree(repo_root, tree_path, max_depth=max_depth)
    _write_project_index(repo_root, index_path, max_depth=max_depth)

    metadata = _collect_basic_metadata(
        repo_root=repo_root,
        snapshot_dir=snapshot_dir,
        tree_path=tree_path,
        index_path=index_path,
    )

    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """
    CLI argument parser for `python -m devtools.generate_context_snapshot`.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate a context snapshot for the dLuxShera repo, including "
            "project tree, project index, and minimal metadata."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help=(
            "Repository root (default: inferred as one level above devtools/ "
            "containing this script)."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Snapshot directory. If omitted, a timestamped folder will be "
            "created under devtools/, e.g. "
            "devtools/context_snapshot_YYYYmmDD-HHMMSS"
        ),
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help=(
            "Maximum directory depth for project tree/index recursion "
            "(default: 4)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    metadata = generate_context_snapshot(
        root=args.root,
        out_dir=args.out,
        max_depth=args.max_depth,
    )

    snapshot_dir = metadata.get("snapshot_dir")
    print(
        f"[generate_context_snapshot] Snapshot written under: {snapshot_dir}"
    )


if __name__ == "__main__":
    main()
