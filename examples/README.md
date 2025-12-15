# Examples

These examples are runnable artifacts (scripts + notebooks). Install the
project in editable mode before executing them:

```bash
python -m pip install -e .
```

## Scripts

Run scripts directly from the repository root; they import code from
`dluxshera.demos`:

```bash
python examples/scripts/run_canonical_astrometry_demo.py --fast --save-plots-dir Results/CanonicalAstrometryDemo
python examples/scripts/run_twoplane_astrometry_demo.py --fast --save-plots-dir Results/TwoplaneAstrometryDemo
```

## Notebooks

Launch Jupyter after installation, select the environment kernel, and open any
notebook under `examples/notebooks/`. The notebooks are written to use the
installed `dluxshera` package—no `sys.path` tweaks should be required.

## Policy

Examples under this directory are not a Python package. Importable logic should
live under `src/dluxshera/` (for example, `dluxshera.demos.canonical_astrometry`),
while the files here stay as runnable references.
