"""Fail if any Python file imports from examples/*.

Examples are runnable artifacts, not packages. This guardrail prevents
``import examples`` or ``import Examples`` statements from creeping into the
codebase.
"""
from __future__ import annotations

import ast
from pathlib import Path

SEARCH_DIRS = ("src", "tests", "devtools", "examples")


def _iter_python_files(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in paths:
        out.extend(path for path in root.rglob("*.py") if path.is_file())
    return out


def _has_disallowed_import(tree: ast.AST) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "examples" or alias.name.startswith("examples."):
                    hits.append((node.lineno, f"import {alias.name}"))
                if alias.name == "Examples" or alias.name.startswith("Examples."):
                    hits.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            if node.module and (
                node.module == "examples"
                or node.module.startswith("examples.")
                or node.module == "Examples"
                or node.module.startswith("Examples.")
            ):
                hits.append((node.lineno, f"from {node.module} import ..."))
    return hits


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    existing_paths = [repo_root / path for path in SEARCH_DIRS if (repo_root / path).exists()]

    if not existing_paths:
        print("No search targets found.")
        return 0

    offenders: list[str] = []
    for file_path in _iter_python_files(existing_paths):
        try:
            source = file_path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError:
            continue
        for lineno, snippet in _has_disallowed_import(tree):
            offenders.append(f"{file_path}:{lineno}: {snippet}")

    if offenders:
        print("Found disallowed imports from examples/:\n")
        print("\n".join(offenders))
        return 1

    print("No imports from examples/ detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
