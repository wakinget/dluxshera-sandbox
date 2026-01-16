# Examples

These examples are runnable artifacts (scripts + notebooks). Install the
project in editable mode before executing them:

```bash
python -m pip install -e .
```

## Recipes + runners

Read-first recipes live under `examples/recipes/`. Execute-first runners live
under `examples/runners/`.

```bash
python examples/runners/run_canonical_astrometry.py --fast
python examples/recipes/twoplane_astrometry.py
python examples/scripts/run_twoplane_astrometry_demo.py --fast --save-plots-dir Results/TwoplaneAstrometryDemo
```

## Notebooks

Launch Jupyter after installation, select the environment kernel, and open any
notebook under `examples/notebooks/`. The notebooks are written to use the
installed `dluxshera` package—no `sys.path` tweaks should be required.

## Policy

Examples under this directory are not a Python package. Keep shared library
logic under `src/dluxshera/`; the files here stay as runnable or read-first
references.
