# Examples

These examples are runnable artifacts (scripts + notebooks). Install the
project in editable mode before executing them:

```bash
python -m pip install -e .
```

## Recipes + runners (start here)

Read-first recipes live under `examples/recipes/`. Execute-first runners live
under `examples/runners/`.

- **Canonical 3-plane runner + recipe** — Shera three-plane end-to-end workflow
  (config → binder → simulate → loss/inference → plotting).
  - Recipe: `examples/recipes/canonical_astrometry.py`
  - Runner: `examples/runners/run_canonical_astrometry.py`
  - How to run:

    ```bash
    python examples/runners/run_canonical_astrometry.py
    python examples/runners/run_canonical_astrometry.py --fast
    ```
- **2-plane runner + recipe** — Same workflow on the two-plane optical system
  for a faster, simpler baseline.
  - Recipe: `examples/recipes/twoplane_astrometry.py`
  - Runner: `examples/runners/run_twoplane_astrometry.py`
  - How to run:

    ```bash
    python examples/runners/run_twoplane_astrometry.py
    python examples/runners/run_twoplane_astrometry.py --fast
    ```

## Notebooks

Launch Jupyter after installation, select the environment kernel, and open any
notebook under `examples/notebooks/`. The notebooks are written to use the
installed `dluxshera` package—no `sys.path` tweaks should be required.

- **minimal_example.ipynb** — Minimal forward-model + optimization walkthrough
  for quick orientation.
- **Shera_Eigen_Inference_Example.ipynb** — Eigen-θ inference in the Shera
  workflow, with plots that mirror the scripts.
- **notebook_setup.py** — Shared notebook utilities (imports, plotting defaults).

To launch Jupyter:

```bash
jupyter lab
```

## Utility scripts

Scripts under `examples/scripts/` operate on run directories produced by the
recipes/runners.

- **summarize_runs.py** — Crawl run directories and emit a CSV summary.
  - How to run:

    ```bash
    python examples/scripts/summarize_runs.py --help
    python examples/scripts/summarize_runs.py Results/
    ```
- **analyze_checkpoint_gradients.py** — Inspect saved checkpoints and write
  gradient diagnostics under a run’s `diag/` directory.
  - How to run:

    ```bash
    python examples/scripts/analyze_checkpoint_gradients.py --help
    python examples/scripts/analyze_checkpoint_gradients.py Results/<run_dir>
    ```

## Artifact outputs (what to look for)

Recipes/runners create a run directory (defaults to `Results/<run_id>/`) that
holds traces, summaries, plots, and optional checkpoints. For a quick tour of
the file layout and optional artifacts, see
`docs/architecture/optimization_artifacts_and_plotting.md`.

## Policy

Examples under this directory are not a Python package. Keep shared library
logic under `src/dluxshera/`; the files here stay as runnable or read-first
references.
