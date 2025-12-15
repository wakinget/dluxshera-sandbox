from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import io
import json
import sys
from collections import defaultdict
from dataclasses import MISSING, is_dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from devtools.introspection import print_tree, build_project_index

# NOTE: The repository uses a ``src/`` layout (package lives under ``src/dluxshera``).
# This module intentionally nudges ``sys.path`` to include both the repo root and the
# ``src/`` directory so that ``python -m devtools.generate_context_snapshot`` works
# regardless of the current working directory. Imports are treated as best-effort;
# failures are recorded in the snapshot metadata instead of aborting the run.


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


def _ensure_repo_on_sys_path(repo_root: Path) -> None:
    """Ensure the repo root (and its ``src`` dir) are importable.

    This keeps introspection imports working when running the script directly via
    ``python -m devtools.generate_context_snapshot`` without an editable install.
    """

    repo_str = str(repo_root)
    src_dir = repo_root / "src"
    src_str = str(src_dir)

    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    if src_dir.exists() and src_str not in sys.path:
        sys.path.insert(0, src_str)


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
        "schema_version": "0.2",
        "generated_at": now.isoformat(timespec="seconds"),
        "repo_root": str(repo_root),
        "snapshot_dir": rel(snapshot_dir),
        "files": {
            "project_tree": rel(tree_path),
            "project_index": rel(index_path),
            "context_snapshot": rel(snapshot_dir / "context_snapshot.json"),
        },
        "notes": [
            "Context snapshot includes project tree, static index, and summarized metadata.",
            "Imports are best-effort; failures are recorded instead of stopping generation.",
        ],
    }

    # Record basic info about the Working Plan, if present.
    working_plan = repo_root / "docs" / "dev" / "dLuxShera_Working_Plan.md"
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


def _summarize_project_index(index: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    """
    Derive compact summary statistics from the project index.
    """

    python_files = 0
    classes = 0
    functions = 0

    def _walk(node: Dict[str, Any]) -> None:
        nonlocal python_files, classes, functions
        if node.get("type") == "file" and node.get("kind") == "python":
            python_files += 1
            classes += len(node.get("classes", []))
            functions += len(node.get("functions", []))
        for child in node.get("children", []) or []:
            _walk(child)

    _walk(index)

    # Discover python packages under src/ with __init__.py present.
    packages: List[str] = []
    src_dir = repo_root / "src"
    if src_dir.exists():
        for init_file in src_dir.rglob("__init__.py"):
            try:
                rel = init_file.relative_to(src_dir)
                dotted = ".".join(rel.parts[:-1])
                if dotted:
                    packages.append(dotted)
            except ValueError:
                continue
    packages = sorted(set(packages))

    return {
        "python_files": python_files,
        "classes": classes,
        "functions": functions,
        "packages": packages,
    }


def _parse_requirements_file(path: Path) -> List[str]:
    deps: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        deps.append(stripped)
    return deps


def _collect_dependencies(repo_root: Path) -> Dict[str, Any]:
    deps: Dict[str, Any] = {"runtime": [], "dev": []}

    # pyproject.toml (optional)
    pyproject = repo_root / "pyproject.toml"
    if pyproject.exists():
        try:
            try:
                import tomllib  # py3.11+
            except ImportError:  # pragma: no cover - fallback
                import tomli as tomllib  # type: ignore

            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            project = data.get("project", {})
            if project.get("dependencies"):
                deps["runtime"].extend(project["dependencies"])
            optional = project.get("optional-dependencies", {})
            for key, values in optional.items():
                if key.lower() in {"dev", "test", "tests", "dev-test"}:
                    deps["dev"].extend(values)
        except Exception as exc:  # pragma: no cover - defensive
            deps["error"] = f"Failed to parse pyproject.toml: {exc!r}"

    req = repo_root / "requirements.txt"
    if req.exists():
        deps["runtime"].extend(_parse_requirements_file(req))

    req_dev = repo_root / "requirements-dev.txt"
    if req_dev.exists():
        deps["dev"].extend(_parse_requirements_file(req_dev))

    # Remove duplicates while preserving order
    def _dedupe(items: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    deps["runtime"] = _dedupe(deps.get("runtime", []))
    deps["dev"] = _dedupe(deps.get("dev", []))
    return deps


def _collect_tests_metadata(repo_root: Path) -> Dict[str, Any]:
    tests_dir = repo_root / "tests"
    if not tests_dir.exists():
        return {}

    import ast

    by_file: Dict[str, Dict[str, Any]] = {}
    by_module: Dict[str, List[str]] = {}

    for test_path in tests_dir.rglob("test_*.py"):
        if not test_path.is_file():
            continue
        try:
            tree = ast.parse(test_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        imports: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name:
                        imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        rel_path = str(test_path.relative_to(repo_root))
        uniq_imports = sorted(set(imports))
        by_file[rel_path] = {"imports": uniq_imports}

        for mod in uniq_imports:
            by_module.setdefault(mod, []).append(rel_path)

    # Sort module file lists for stability
    by_module = {k: sorted(v) for k, v in sorted(by_module.items())}

    return {"by_file": by_file, "by_module": by_module}


def _summarize_param_spec(spec: Any) -> Dict[str, Any]:
    """Compact representation of ParamSpec contents."""

    try:
        fields_iter = list(spec.values()) if hasattr(spec, "values") else []
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": f"Failed to iterate spec: {exc!r}"}

    total = len(fields_iter)
    by_kind: Dict[str, int] = {}
    by_group: Dict[str, int] = {}
    primitives: List[str] = []
    derived: List[str] = []
    others: List[str] = []
    derived_dependencies: List[Dict[str, Any]] = []

    for field in fields_iter:
        key = getattr(field, "key", None)
        kind = getattr(field, "kind", "unknown") or "unknown"
        group = getattr(field, "group", None)

        by_kind[kind] = by_kind.get(kind, 0) + 1
        if group:
            by_group[group] = by_group.get(group, 0) + 1

        if key:
            if kind == "primitive":
                primitives.append(key)
            elif kind == "derived":
                derived.append(key)
            else:
                others.append(key)

        depends_on = getattr(field, "depends_on", ()) or ()
        if kind == "derived":
            derived_dependencies.append(
                {"key": key, "depends_on": list(depends_on)}
            )

    def _sample(values: List[str], *, limit: int = 10) -> List[str]:
        return values[:limit]

    return {
        "count": total,
        "primitive_count": len(primitives),
        "derived_count": len(derived),
        "other_count": len(others),
        "kinds": by_kind,
        "groups": by_group,
        "primitive_keys_sample": _sample(primitives),
        "derived_keys_sample": _sample(derived),
        "other_keys_sample": _sample(others),
        "derived_dependencies": derived_dependencies,
    }


def _collect_param_specs(repo_root: Path) -> Dict[str, Any]:
    _ensure_repo_on_sys_path(repo_root)
    try:
        from dluxshera.params import spec as params_spec
        from dluxshera.optics.config import SheraThreePlaneConfig, SheraTwoPlaneConfig
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": f"Failed to import param spec modules: {exc!r}"}

    specs: List[Dict[str, Any]] = []

    spec_builders: Iterable[Dict[str, Any]] = [
        {
            "name": "inference_basic",
            "builder": params_spec.build_inference_spec_basic,
            "kwargs": {},
            "system_id": "shera_threeplane",
            "category": "inference",
        },
        {
            "name": "forward_threeplane",
            "builder": params_spec.build_forward_model_spec_from_config,
            "kwargs": {"cfg": SheraThreePlaneConfig()},
            "system_id": "shera_threeplane",
            "category": "forward",
        },
        {
            "name": "forward_twoplane",
            "builder": params_spec.build_shera_twoplane_forward_spec_from_config,
            "kwargs": {"cfg": SheraTwoPlaneConfig()},
            "system_id": "shera_twoplane",
            "category": "forward",
        },
    ]

    for entry in spec_builders:
        spec_meta: Dict[str, Any] = {
            "name": entry.get("name"),
            "system_id": entry.get("system_id"),
            "category": entry.get("category"),
        }
        try:
            builder_fn = entry["builder"]
            kwargs = entry.get("kwargs", {})
            spec_obj = builder_fn(**kwargs)
            spec_meta.update(_summarize_param_spec(spec_obj))
        except Exception as exc:  # pragma: no cover - defensive
            spec_meta["error"] = f"{exc!r}"
        specs.append(spec_meta)

    systems: Dict[str, Any] = {}
    for spec in specs:
        sid = spec.get("system_id") or "unspecified"
        systems.setdefault(sid, {"spec_names": [], "count": 0})
        systems[sid]["spec_names"].append(spec.get("name"))
        systems[sid]["count"] += 1

    return {"modules": ["dluxshera.params.spec"], "specs": specs, "systems": systems}


def _collect_transforms(repo_root: Path) -> Dict[str, Any]:
    _ensure_repo_on_sys_path(repo_root)
    try:
        from dluxshera.params import registry as registry_mod
        from dluxshera.params import transforms  # Registers transforms on import
        from dluxshera.params import shera_threeplane_transforms  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": f"Failed to import transforms: {exc!r}"}

    try:
        resolver = getattr(transforms, "DERIVED_RESOLVER", None)
        if resolver is None:
            resolver = registry_mod.DerivedResolver(default_system_id="default")

        systems: Dict[str, Any] = {}
        registries = getattr(resolver, "_registries", {})

        for system_id, registry in sorted(registries.items()):
            transform_map = getattr(registry, "_transforms", {})
            entries = []
            for key, transform in sorted(transform_map.items()):
                doc = (getattr(transform, "doc", None) or "").strip() or None
                if doc and len(doc) > 200:
                    doc = doc[:197] + "…"
                entries.append(
                    {
                        "key": key,
                        "depends_on": list(getattr(transform, "depends_on", ())),
                        "doc": doc,
                    }
                )

            systems[system_id] = {
                "count": len(entries),
                "transform_keys": [e.get("key") for e in entries],
                "transforms": entries,
            }

        return {
            "modules": [
                "dluxshera.params.transforms",
                "dluxshera.params.shera_threeplane_transforms",
            ],
            "systems": systems,
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": f"Failed to introspect transform registry: {exc!r}"}


def _format_default(value: Any, max_length: int = 120) -> str:
    repr_val = repr(value)
    if len(repr_val) > max_length:
        return repr_val[: max_length - 1] + "…"
    return repr_val


def _collect_configs_metadata(repo_root: Path) -> Dict[str, Any]:
    _ensure_repo_on_sys_path(repo_root)
    try:
        from dluxshera import optics
        from dluxshera.optics import builder
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": f"Failed to import optics configs: {exc!r}"}

    candidate_names = ["SheraTwoPlaneConfig", "SheraThreePlaneConfig"]
    cfg_classes = [
        getattr(optics.config, name)
        for name in candidate_names
        if hasattr(optics.config, name)
    ]

    configs: List[Dict[str, Any]] = []
    for cls in cfg_classes:
        try:
            instance = cls()
        except Exception:
            instance = None

        field_summaries = []
        for f in fields(cls):
            default_factory = getattr(f, "default_factory", MISSING)
            has_default = f.default is not MISSING
            has_factory = default_factory is not MISSING  # type: ignore[comparison-overlap]
            default_repr = None
            if instance is not None:
                default_repr = _format_default(getattr(instance, f.name))
            elif has_default:
                default_repr = _format_default(f.default)
            elif has_factory:
                default_repr = _format_default(default_factory)

            field_summaries.append(
                {
                    "name": f.name,
                    "type": getattr(f.type, "__name__", str(f.type)),
                    "has_default": default_repr is not None,
                    "default_repr": default_repr,
                }
            )

        cfg_meta: Dict[str, Any] = {
            "type": cls.__name__,
            "module": cls.__module__,
            "field_count": len(field_summaries),
            "fields": field_summaries,
        }

        if instance is not None:
            try:
                if hasattr(builder, "structural_hash_from_config") and isinstance(
                    instance, getattr(optics.config, "SheraThreePlaneConfig", object)
                ):
                    cfg_meta["structural_hash_example"] = builder.structural_hash_from_config(instance)
                if hasattr(builder, "structural_hash_for_twoplane") and isinstance(
                    instance, getattr(optics.config, "SheraTwoPlaneConfig", object)
                ):
                    cfg_meta["structural_hash_example"] = builder.structural_hash_for_twoplane(instance)
            except Exception:
                pass

        configs.append(cfg_meta)

    return {
        "modules": ["dluxshera.optics.config"],
        "configs": configs,
    }


def _collect_demo_metadata(repo_root: Path) -> List[Dict[str, Any]]:
    demos = [
        {
            "script": "examples/scripts/run_canonical_astrometry_demo.py",
            "system_id": "shera_threeplane",
            "param_specs": ["forward_threeplane", "inference_basic"],
        },
        {
            "script": "examples/scripts/run_twoplane_astrometry_demo.py",
            "system_id": "shera_twoplane",
            "param_specs": ["forward_twoplane", "inference_basic"],
        },
    ]

    for entry in demos:
        script_path = repo_root / entry["script"]
        entry["exists"] = script_path.exists()

    return demos


def _generate_markdown_report(meta: Dict[str, Any], path: Path) -> None:
    """Create a concise human-readable Markdown report."""

    summary = meta.get("summary", {})
    param_specs = meta.get("param_specs", {})
    transforms = meta.get("transforms", {})
    configs = meta.get("configs", {}).get("configs", []) if isinstance(meta.get("configs", {}), dict) else []
    demos = meta.get("demos", [])

    lines = [
        f"# Context Snapshot ({meta.get('generated_at')})",
        "",
        "## Summary",
        f"- Python files: {summary.get('python_files', 'n/a')}",
        f"- Classes: {summary.get('classes', 'n/a')}  Functions: {summary.get('functions', 'n/a')}",
        f"- Packages: {', '.join(summary.get('packages', [])[:8]) if summary.get('packages') else 'n/a'}",
        "",
    ]

    wp = meta.get("working_plan", {})
    if wp.get("exists"):
        lines.extend(
            [
                "## Working Plan",
                f"- Path: {wp.get('path')}",
                f"- Size: {wp.get('size_bytes')} bytes",
                f"- Modified: {wp.get('mtime_iso')}",
                "",
            ]
        )

    if isinstance(param_specs, dict) and param_specs.get("specs"):
        lines.append("## Param Specs")
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for spec in param_specs.get("specs", []):
            grouped[spec.get("system_id") or "unspecified"].append(spec)

        for system_id in sorted(grouped):
            system_specs = grouped[system_id]
            lines.append(f"- System: {system_id} ({len(system_specs)} specs)")
            for spec in system_specs:
                if "error" in spec:
                    lines.append(f"  - {spec.get('name')}: ERROR - {spec['error']}")
                    continue

                primitive_count = spec.get("primitive_count", 0)
                derived_count = spec.get("derived_count", 0)
                lines.append(
                    f"  - {spec.get('name')}: {spec.get('count', 'n/a')} keys "
                    f"({primitive_count} primitive, {derived_count} derived)"
                )

                primaries = spec.get("primitive_keys_sample") or []
                if primaries:
                    ellipsis = " …" if primitive_count > len(primaries) else ""
                    lines.append(
                        f"    • primitives: {', '.join(primaries)}{ellipsis}"
                    )

                derived_entries = spec.get("derived_dependencies", [])
                if derived_entries:
                    formatted = []
                    for entry in derived_entries[:6]:
                        deps = entry.get("depends_on") or []
                        dep_str = ", ".join(deps) if deps else "none"
                        formatted.append(f"{entry.get('key')}: {dep_str}")
                    remainder = "" if len(derived_entries) <= 6 else " …"
                    lines.append(f"    • derived: { '; '.join(formatted)}{remainder}")

                groups = spec.get("groups") or {}
                if groups:
                    group_str = ", ".join(
                        f"{name}={count}" for name, count in sorted(groups.items())
                    )
                    lines.append(f"    • groups: {group_str}")
        lines.append("")
    elif isinstance(param_specs, dict) and param_specs.get("error"):
        lines.extend(["## Param Specs", f"- ERROR: {param_specs['error']}", ""])

    if isinstance(transforms, dict) and transforms.get("systems"):
        lines.append("## Transforms")
        systems = transforms.get("systems", {})
        for sid, info in sorted(systems.items()):
            lines.append(f"- System: {sid} ({info.get('count', 'n/a')} transforms)")
            for entry in info.get("transforms", []):
                deps = entry.get("depends_on") or []
                dep_str = ", ".join(deps) if deps else "none"
                doc = entry.get("doc")
                doc_str = f" – {doc}" if doc else ""
                lines.append(
                    f"  - {entry.get('key')}: depends on {dep_str}{doc_str}"
                )
        lines.append("")
    elif isinstance(transforms, dict) and "error" in transforms:
        lines.extend(["## Transforms", f"- ERROR: {transforms['error']}", ""])

    if configs:
        lines.append("## Configs")
        for cfg in configs:
            lines.append(f"- {cfg.get('type')}: {len(cfg.get('fields', []))} fields")
        lines.append("")

    if demos:
        lines.append("### Demos and ParamSpecs")
        for demo in demos:
            status = "present" if demo.get("exists") else "missing"
            specs = ", ".join(demo.get("param_specs", []))
            lines.append(
                f"- {demo.get('script')}: system {demo.get('system_id')} → {specs} ({status})"
            )
        lines.append("")

    content = "\n".join(lines).rstrip() + "\n"
    path.write_text(content, encoding="utf-8")


def generate_context_snapshot(
    *,
    root: Optional[str] = None,
    out_dir: Optional[str] = None,
    max_depth: int = 4,
    generate_markdown: bool = True,
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
    _ensure_repo_on_sys_path(repo_root)
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
    md_path = snapshot_dir / "context_snapshot.md"

    _write_project_tree(repo_root, tree_path, max_depth=max_depth)
    _write_project_index(repo_root, index_path, max_depth=max_depth)

    metadata = _collect_basic_metadata(
        repo_root=repo_root,
        snapshot_dir=snapshot_dir,
        tree_path=tree_path,
        index_path=index_path,
    )

    try:
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        metadata["summary"] = _summarize_project_index(index_data, repo_root)
    except Exception as exc:  # pragma: no cover - defensive
        metadata["summary"] = {"error": f"Failed to summarize project index: {exc!r}"}

    metadata["dependencies"] = _collect_dependencies(repo_root)
    metadata["tests"] = _collect_tests_metadata(repo_root)
    metadata["param_specs"] = _collect_param_specs(repo_root)
    metadata["transforms"] = _collect_transforms(repo_root)
    metadata["configs"] = _collect_configs_metadata(repo_root)
    metadata["demos"] = _collect_demo_metadata(repo_root)

    if generate_markdown:
        try:
            _generate_markdown_report(metadata, md_path)
            metadata.setdefault("files", {})["context_snapshot_md"] = str(
                md_path.relative_to(repo_root)
            )
        except Exception as exc:  # pragma: no cover - defensive
            metadata.setdefault("notes", []).append(
                f"Failed to write Markdown report: {exc!r}"
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
            "project tree, static index, and enriched metadata."
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
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip generation of the human-readable context_snapshot.md report.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    metadata = generate_context_snapshot(
        root=args.root,
        out_dir=args.out,
        max_depth=args.max_depth,
        generate_markdown=not args.no_markdown,
    )

    snapshot_dir = metadata.get("snapshot_dir")
    print(
        f"[generate_context_snapshot] Snapshot written under: {snapshot_dir}"
    )


if __name__ == "__main__":
    main()
