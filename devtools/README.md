# dLuxShera Devtools

This folder contains developer-oriented utilities for inspecting, documenting, 
and generating context snapshots for the dLuxShera codebase. These tools are not 
part of the installed `dluxshera` package but are used during development and 
for interacting with ChatGPT/Codex.

The devtools package includes:

- `print_tree.py`  
  Prints a directory tree and optionally produces a JSON project index.

- `introspection.py`  
  Shared helper utilities for AST parsing and project indexing.

- `generate_context_snapshot.py`  
  Generates a context snapshot directory containing:
  - `project_tree.txt` (ASCII project tree)
  - `project_index.json` (static code index: modules, classes, functions)
  - `context_snapshot.json` (metadata, including Working Plan info)

  The format is designed to be extended later with ParamSpec summaries,
  transform registry state, SystemGraph topology, and test coverage maps.

- `context_snapshot_<timestamp>/`  
  Auto-generated folders containing project tree, JSON index, and extended 
  context metadata. These are ignored by Git.

---

## 1. Running Devtools Scripts

All devtools scripts should be run using the module form:

```bash
python -m devtools.print_tree
```

```bash
python -m devtools.generate_context_snapshot
```

Running scripts this way ensures correct import paths and consistent behavior.

---

## 2. `print_tree.py` Usage

Summary of common operations.

### Basic usage:

```bash
python -m devtools.print_tree
```

### Set a custom root:

```bash
python -m devtools.print_tree --root src/dluxshera
```

### Control recursion depth:

```bash
python -m devtools.print_tree --max-depth 5
```

### Save a JSON index:

```bash
python -m devtools.print_tree --json-out devtools/project_index.json
```

### Save printed tree to a file:

```bash
python -m devtools.print_tree --json-out devtools/project_index.json \
    > devtools/project_tree.txt
```

The JSON index describes modules, classes, functions, signatures, and docstring 
summaries.

---

## 3. `generate_context_snapshot.py`

This tool generates a self-contained snapshot of the repo layout and code index
for use with ChatGPT/Codex.

### 3.1 Current behavior

Running:

```bash
python -m devtools.generate_context_snapshot
```

will:

- Infer the repo root as one level above `devtools/`.
- Create a timestamped snapshot directory under `devtools/`, e.g.:

```text
devtools/context_snapshot_20251211-162732/
```

- Write the following files into that directory:

  - `project_tree.txt`  
    ASCII tree view of the repository (same content as `print_tree.py` output).

  - `project_index.json`  
    Nested JSON index describing directories/files and, for `.py` files,
    modules, classes, functions, signatures, and docstring summaries.

  - `context_snapshot.json`  
    Metadata describing the snapshot:
    - `schema_version`, `generated_at`, `repo_root`, `snapshot_dir`
    - a `files` section with paths to `project_tree` and `project_index`
    - basic info about `dLuxShera_Refactor_Working_Plan.md` if present
      (relative path, size in bytes, modification time)

CLI options:

```bash
python -m devtools.generate_context_snapshot --help
```

Key arguments:

- `--root PATH`  
  Override the inferred repo root (default: one level above `devtools/`).

- `--out PATH`  
  Custom snapshot directory. If relative, it is interpreted relative to the
  repo root. If omitted, a timestamped folder under `devtools/` is used.

- `--max-depth N`  
  Maximum directory depth for the tree and index (default: 4).

### 3.2 Planned extensions

The snapshot format is intentionally minimal for now. Future extensions may add:

- ParamSpec summaries (keys, primitive/derived flags, forward/inference subsets)
- Transform registry state and DerivedResolver coverage
- Config defaults and structural hashes for optics builders
- Binder/SystemGraph wiring maps
- Simple test → module coverage hints
- An optional Markdown report (`context_snapshot.md`) that summarises the above
  for quick human review

---

## 4. Snapshot Directory Conventions

Inside a snapshot folder, the following files may appear:

- `project_tree.txt`  
- `project_index.json`  
- `context_snapshot.json`  
- `context_snapshot.md` (optional)

These files are generated on demand and provide a consistent context bundle for 
development, debugging, and AI-assisted workflows.

---

## 5. Developer Notes

- The `devtools/` directory *is* version-controlled.  
- Snapshot folders under `devtools/context_snapshot_*/` are *not* tracked.  
- Shared logic lives in `introspection.py`.  
- Scripts under `devtools/` should remain independent of the primary 
  `src/dluxshera` package for clean separation of concerns.

