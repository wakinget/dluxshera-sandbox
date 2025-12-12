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
  - `context_snapshot.json` (metadata: working plan info, summary counts,
    dependencies, tests, ParamSpecs, transforms, configs)
  - `context_snapshot.md` (human-readable overview)

  The format now includes ParamSpec summaries, transform registry state, optics
  configs, and simple test coverage maps in addition to the static tree/index.

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

- `context_snapshot.json` (schema_version `0.2`)
    Metadata describing the snapshot. In addition to paths and working plan
    info it includes:
    - `summary`: Python file/class/function counts and discovered packages
    - `dependencies`: runtime/dev requirements parsed from pyproject/requirements
    - `tests`: imports observed in test files (by file and by module)
    - `param_specs`: compact summaries of baseline ParamSpecs
    - `transforms`: registered derived parameter transforms
    - `configs`: dataclass fields and example structural hashes for optics configs

- `context_snapshot.md`
    Lightweight Markdown overview for quick human review (optional; enabled by
    default).

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

- `--no-markdown`
  Skip generation of `context_snapshot.md`.

### 3.2 Planned extensions

Future extensions may add:

- Binder/SystemGraph wiring maps
- Additional system builder summaries
- Richer coverage links between tests and modules

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

