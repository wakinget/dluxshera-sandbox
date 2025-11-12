# dLuxShera

High-precision astrometry & PSF modeling on top of dLux (JAX). This repo contains a three-plane optical model (M1/M2/Detector) and inference utilities.

## TL;DR
- **Goal:** recover plate scale + optical aberrations from stellar images (e.g., SHERA/TOLIMAN-like setups).
- **Status:** working model + example scripts; docs are growing.
- **Start here:** Quickstart below, then see docs for options like eigenmode truncation.

## Install (dev)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"   # if you have extras; otherwise: pip install -e .
```

> Requires Python ≥3.10 and JAX/dLux. CPU is fine; GPU not required.

## Quickstart
```python
from dLuxShera.optical_systems import SheraThreePlane_Model  # adjust if your name differs
# Minimal example — build a model with defaults and generate a PSF
model = SheraThreePlane_Model()
psf = model()                      # or model.forward(), depending on your API
print(psf.shape)
```

## Common tasks
- **Astrometric fit:** `python scripts/run_fit.py --config configs/acen.yaml`
- **Sweep eigenmodes:** `python scripts/run_eigen_sweep.py --truncate_by_eigval 1e4`
- **Visualize PSF:** `python scripts/show_psf.py --wavelength 550e-9`

## Key concepts
- **Eigen re-parameterization:** `use_eigen`, `whiten_basis`, `truncate_k`, `truncate_by_eigval`
- **Plate scale:** `psf_pixel_scale` can be optimized directly or derived from primitives.
- **Numerics:** Fresnel propagation via custom three-plane routine (dLux-based, JAX grad-safe).

## Troubleshooting
- **Zero gradients from “perfect” init:** initialize with *non-zero* tiny perturbations.
- **Slow JIT on first call:** expected; subsequent runs are fast.
- **Diffs not showing?** `git diff -- path/to/file.py` or `git diff --staged`.

## Documentation
Local docs preview:
```bash
pip install mkdocs mkdocs-material "mkdocstrings[python]" griffe
mkdocs serve
```

## Contributing
- `pip install -r requirements-dev.txt` (black, isort, pytest, pre-commit)
- `pre-commit install`
- Branch: `feature/<topic>`; PRs welcome.
