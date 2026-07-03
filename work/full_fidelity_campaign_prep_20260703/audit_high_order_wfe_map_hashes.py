from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


MAP_SUFFIXES = {".npy", ".npz", ".fits", ".json", ".yaml", ".yml"}
MAP_NAME_TOKENS = ("high_order", "wfe", "opd", "map")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_files(run_root: Path) -> list[Path]:
    model_split = run_root / "model_split"
    root = model_split if model_split.exists() else run_root
    out: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in MAP_SUFFIXES:
            continue
        lowered = path.as_posix().lower()
        if any(token in lowered for token in MAP_NAME_TOKENS):
            out.append(path)
    return sorted(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hash high-order WFE map artifacts from dry-run/preflight run roots."
    )
    parser.add_argument("run_roots", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for run_root in args.run_roots:
        for path in candidate_files(run_root):
            rows.append(
                {
                    "run_root": str(run_root),
                    "relative_path": path.relative_to(run_root).as_posix(),
                    "sha256": sha256(path),
                    "bytes": str(path.stat().st_size),
                }
            )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("run_root", "relative_path", "sha256", "bytes")
            )
            writer.writeheader()
            writer.writerows(rows)
    writer = csv.DictWriter(
        __import__("sys").stdout,
        fieldnames=("run_root", "relative_path", "sha256", "bytes"),
    )
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
