# dLuxShera

High-precision, differentiable astrometric instrument model using dLux (JAX) for auto-diff. This repo contains prescribed 3-plane optical models for the SHERA telescope and inference utilities. This repo allows users to initialize an optical system and set up parameter inference optimization loops 

## TL;DR
- **Overview**: Build and simulate a differentiable three-plane Fresnel optical model (SHERA).
- **Goal**: Recover binary separation, plate scale, and low-order WFE from synthetic images.
- **Status**: Core model + inference utilities are functional; examples and docs in progress.
- **Start here**: See the Quickstart below, then explore the example notebook.

## Install

This repository is not yet an installable Python package. Installation is done by:

1. Creating a virtual environment  
2. Installing dependencies from `requirements.txt`  
3. Running notebooks/scripts from within that environment  

This will automatically install a temporary **Fresnel-enabled fork of dLux** until the upstream PR is merged.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Upgrade pip (recommended)
pip install --upgrade pip

# 3. Install dependencies
pip install -r requirements.txt
```

After installation, your environment includes:

- JAX + jaxlib  
- Your Fresnel-enabled dLux fork (via PEP 508 URL)  
- numpy, scipy, matplotlib, astropy  
- numpyro, optax, equinox, chex, jaxtyping, zodiax  
- JupyterLab  

> **Note:** Once the Fresnel PR merges into upstream dLux, this step will be updated to rely on a released version.

---

## Quickstart (Jupyter Notebook)

An example notebook is provided at:

```
Examples/notebooks/Shera_Eigen_Inference_Example.ipynb
```

Launch JupyterLab **from inside the virtual environment**:

```bash
source .venv/bin/activate
jupyter lab
```

Then open the example notebook in the sidebar.

### Notebook setup

Each notebook begins with a path helper that automatically locates the repository root, ensuring imports work no matter where Jupyter was launched. This should happen automatically without user input.

```python
import notebook_setup
repo_root = notebook_setup.setup_paths()
```

### Minimal example — build a model and generate a PSF

```python
# Notebook path setup
import notebook_setup
repo_root = notebook_setup.setup_paths()

# Model + utilities
from Classes.modeling import SheraThreePlane_Model

# Build model with default parameters
model = SheraThreePlane_Model()

# Forward simulation (PSF image)
psf = model.model()
```

This produces a simulated PSF image using the default three-plane SHERA optical model.

---

### Notes

- The new Fresnel propagation utilities for dLux are currently under review. For the time being, dLux installation uses my own local fork. When the PR is fully integrated, these installation instructions will change.
- Notebooks rely on `notebook_setup.py` located in `Examples/notebooks/`.  
- The repo is not yet a Python package; imports follow the current directory structure:
  ```
  from Classes.modeling import SheraThreePlane_Model
  from Classes.optimization import ...
  ```
- A `pyproject.toml` will be added later to support `pip install -e .`.


## Key concepts
- **Eigen re-parameterization:** `use_eigen`, `whiten_basis`, `truncate_k`, `truncate_by_eigval`
- **Plate scale:** `psf_pixel_scale` can be optimized directly or derived from primitives.
- **Numerics:** Fresnel propagation via custom three-plane routine (dLux-based, JAX grad-safe).

## Troubleshooting
- **Zero gradients from “perfect” init:** initialize with *non-zero* tiny perturbations.
- **Slow JIT on first call:** expected; subsequent runs are fast.
- **Diffs not showing?** `git diff -- path/to/file.py` or `git diff --staged`.

## Documentation
(MkDocs configuration pending; this command will serve once docs are added.)
