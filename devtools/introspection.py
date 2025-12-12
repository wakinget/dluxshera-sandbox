from __future__ import annotations

from pathlib import Path
import ast
import json
from typing import Any, Dict, List, Optional

# Reuse the exclude rules used by print_tree
EXCLUDE_DIRS = {
    ".git", ".idea", ".vscode", "__pycache__", "venv", ".venv",
    ".mypy_cache", ".pytest_cache", "Results",
}
EXCLUDE_FILES = {".DS_Store"}


# ---------------------------------------------------------------------------
# Tree-printing functionality
# ---------------------------------------------------------------------------

def print_tree(root: Path, max_depth: int = 4, prefix: str = ""):
    """
    Recursively print a simple tree of the project.

    This preserves all behavior from print_tree.py, including the Examples/
    special case.
    """
    root = Path(root)

    def _recurse(path: Path, depth: int, prefix: str):
        if depth > max_depth:
            return

        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        for i, entry in enumerate(entries):
            if entry.name in EXCLUDE_FILES:
                continue
            if entry.is_dir() and entry.name in EXCLUDE_DIRS:
                continue
            # Skip auto-generated context snapshot dirs (e.g. context_snapshot_YYYYmmDD-HHMMSS)
            if entry.is_dir() and entry.name.startswith("context_snapshot_"):
                continue

            is_last = (i == len(entries) - 1)
            connector = "└── " if is_last else "├── "
            print(prefix + connector + entry.name)

            # Descend into directories
            if entry.is_dir():
                new_prefix = prefix + ("    " if is_last else "│   ")

                # Special-case: Examples → only index notebooks/
                if entry.name == "Examples":
                    notebooks = entry / "notebooks"
                    if notebooks.exists() and notebooks.is_dir():
                        print(new_prefix + "└── notebooks")
                        _recurse(notebooks, depth + 1, new_prefix + "    ")
                    continue

                _recurse(entry, depth + 1, new_prefix)

    print(root.name + "/")
    _recurse(root, depth=1, prefix=prefix)


# ---------------------------------------------------------------------------
# Code indexing helpers
# ---------------------------------------------------------------------------

def _summarize_docstring(doc: Optional[str], max_len: int = 80) -> Optional[str]:
    """Return a one-line summary of a docstring (first line, truncated)."""
    if not doc:
        return None
    first = doc.strip().splitlines()[0].strip()
    if len(first) > max_len:
        return first[: max_len - 1] + "…"
    return first


def _format_arguments(args: ast.arguments) -> str:
    """Reconstruct a Python-like signature from an ast.arguments object."""
    parts: List[str] = []

    posonly = getattr(args, "posonlyargs", []) or []
    regular = list(args.args or [])
    all_pos = list(posonly) + list(regular)

    defaults = list(args.defaults or [])
    num_no_default = len(all_pos) - len(defaults)

    # Positional & defaults
    for i, arg in enumerate(all_pos):
        name = arg.arg
        if i >= num_no_default:
            default_node = defaults[i - num_no_default]
            try:
                default_repr = ast.unparse(default_node)
            except Exception:
                default_repr = "..."
            parts.append(f"{name}={default_repr}")
        else:
            parts.append(name)

        if posonly and i + 1 == len(posonly):
            parts[-1] = parts[-1] + " /"

    # *args
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")

    # Keyword-only
    if args.kwonlyargs:
        if args.vararg is None:
            parts.append("*")
        for kwarg, default in zip(args.kwonlyargs, args.kw_defaults):
            if default is None:
                parts.append(kwarg.arg)
            else:
                try:
                    default_repr = ast.unparse(default)
                except Exception:
                    default_repr = "..."
                parts.append(f"{kwarg.arg}={default_repr}")

    # **kwargs
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")

    return "(" + ", ".join(parts) + ")"


def index_python_file(path: Path) -> Dict[str, Any]:
    """
    Build a summary for a single Python file: classes, methods, functions.
    """
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(path))

    module_doc = _summarize_docstring(ast.get_docstring(module))
    classes = []
    functions = []

    for node in module.body:
        if isinstance(node, ast.ClassDef):
            class_doc = _summarize_docstring(ast.get_docstring(node))
            methods = []

            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sig = _format_arguments(item.args)
                    meth_doc = _summarize_docstring(ast.get_docstring(item))
                    methods.append(
                        {
                            "name": item.name,
                            "signature": sig,
                            "doc": meth_doc,
                            "async": isinstance(item, ast.AsyncFunctionDef),
                        }
                    )

            classes.append({"name": node.name, "doc": class_doc, "methods": methods})

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = _format_arguments(node.args)
            func_doc = _summarize_docstring(ast.get_docstring(node))
            functions.append(
                {
                    "name": node.name,
                    "signature": sig,
                    "doc": func_doc,
                    "async": isinstance(node, ast.AsyncFunctionDef),
                }
            )

    return {
        "type": "file",
        "kind": "python",
        "name": path.name,
        "path": str(path),
        "module_doc": module_doc,
        "classes": classes,
        "functions": functions,
    }


def build_project_index(root: Path, max_depth: int = 4) -> Dict[str, Any]:
    """
    Build a nested JSON-friendly index with class/function metadata.
    Mirrors print_tree's traversal and special casing for Examples/.
    """
    root = Path(root)

    def _recurse(path: Path, depth: int) -> Dict[str, Any]:
        if depth > max_depth:
            return {"type": "directory", "name": path.name, "path": str(path), "children": []}

        if path.is_dir():
            node = {"type": "directory", "name": path.name, "path": str(path), "children": []}
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))

            for entry in entries:
                if entry.name in EXCLUDE_FILES:
                    continue
                if entry.is_dir() and entry.name in EXCLUDE_DIRS:
                    continue
                # Skip auto-generated context snapshot dirs (e.g. context_snapshot_YYYYmmDD-HHMMSS)
                if entry.is_dir() and entry.name.startswith("context_snapshot_"):
                    continue

                if entry.is_dir() and entry.name == "Examples":
                    notebooks = entry / "notebooks"
                    if notebooks.exists():
                        node["children"].append(_recurse(notebooks, depth + 1))
                    continue

                if entry.is_dir():
                    node["children"].append(_recurse(entry, depth + 1))
                else:
                    if entry.suffix == ".py":
                        node["children"].append(index_python_file(entry))
                    else:
                        node["children"].append(
                            {"type": "file", "kind": "other", "name": entry.name, "path": str(entry)}
                        )

            return node

        # File (rare root case)
        return index_python_file(path) if path.suffix == ".py" else {
            "type": "file", "kind": "other", "name": path.name, "path": str(path)
        }

    return _recurse(root, depth=0)


def save_project_index_json(root: Path, out_path: Path, max_depth: int = 4):
    """
    Convenience wrapper to write project index JSON.
    """
    index = build_project_index(root, max_depth=max_depth)
    Path(out_path).write_text(json.dumps(index, indent=2), encoding="utf-8")
