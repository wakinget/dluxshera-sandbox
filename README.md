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

After installation, your environment should include:

- JAX + jaxlib  
- The Fresnel-enabled dLux (via Dylan's own [fork of dLux](https://github.com/wakinget/dLux))  
- numpy, scipy, matplotlib, astropy  
- numpyro, optax, equinox, chex, jaxtyping, zodiax  
- JupyterLab  

> **Note:** Once the Fresnel utilities officially merge into upstream dLux, this step will be updated to rely on the released version.

---

##  Quickstart for Non-Python Users

If you are not a regular Python user, don’t worry — this project is designed to be easy to set up.  
Below is the recommended workflow using **Miniconda** (a lightweight Python distribution) and **VS Code** (a friendly editor).  
This avoids modifying your system Python and keeps everything clean and isolated.

---

### 1. Install Miniconda (Python 3.11)

1. Download Miniconda from: https://docs.conda.io/en/latest/miniconda.html  
2. Run the installer with default settings.  
3. Open a new terminal:
   - **Windows**: Anaconda Prompt  
   - **macOS/Linux**: regular terminal  

Verify installation with:

```
conda --version
```

---

### 2. Create a clean environment for the model

```
conda create -n dLuxShera python=3.11
conda activate dLuxShera
```

This keeps everything isolated and easy to delete later.

---

### 3. Clone the repository and install dependencies

```
cd <folder-where-you-want-projects>
git clone <REPO_URL>
cd <REPO_FOLDER>

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This installs all required libraries, including the correct pinned dLux commit.

---

### 4. Install Visual Studio Code (VS Code)

Download from: https://code.visualstudio.com/

Open the project:

```
code .
```

If prompted, install the **Python extension**.

Select the correct interpreter:

1. Ctrl/Cmd+Shift+P → “Python: Select Interpreter”  
2. Choose: **Python 3.11 (conda env: dLuxShera)**

---

### 5. Verify the installation

```
python -c "import jax, dLux; print('JAX:', jax.__version__); print('dLux:', dLux.__version__)"
```

If versions print without errors, your setup is working.

Try running one of the example scripts or notebooks next.

---

### 6. Daily usage (the only two commands you need)

```
conda activate dLuxShera
cd <REPO_FOLDER>
code .
```

This opens both the environment and the project.

---

### 7. Optional: Using Jupyter notebooks

If you open a `.ipynb` file:

- Select the **dLuxShera** kernel in the top-right.
- Run cells with Shift+Enter.

---

### That’s it!

Once Miniconda and VS Code are installed, everything should “just work.”  
If something breaks, simply delete and recreate the `dLuxShera` environment — conda environments are disposable.


---

## Quickstart (Jupyter Notebook)

Example notebooks are provided under: `Examples/notebooks/`

To run any of the examples, launch JupyterLab **from inside the virtual environment**:

```bash
source .venv/bin/activate # Activate your virtual environment
jupyter lab # Start up a Jupyter lab session (opens in browser)
```

Then open the example notebook from the sidebar.

### Notebook setup

Each notebook begins with a path helper that automatically locates the repository root, ensuring imports work no matter where Jupyter was launched. This should happen automatically without user input.

```python
import notebook_setup
repo_root = notebook_setup.setup_paths()
```

### Minimal example — build a model and generate a PSF

`minimal_example.ipynb`

A lightweight introduction to the SHERA three-plane optical model.

This notebook walks through constructing the default model, evaluating the forward simulation, and visualizing the resulting polychromatic PSF. It is intended as the quickest way to confirm that the installation works and to understand the basic model workflow.

### Eigenmode Inference — recovers parameters as eigenmodes

`Shera_Eigen_Inference_Example.ipynb`

A full demonstration of parameter retrieval using the eigenmode-based optimization pipeline.

The notebook simulates synthetic data, initializes the SHERA model, computes its Fisher-information eigenbasis, and re-parameterizes the model in terms of these eigenmodes. An iterative optimization loop then solves for the eigenmode coefficients, recovering the underlying physical parameters. Diagnostic plots and convergence summaries are generated throughout.

## Canonical astrometry demo (refactor stack)

The refactor-native, end-to-end astrometry demo lives in `Examples/scripts/run_canonical_astrometry_demo.py`.

Run it directly from the repository root:

```bash
python -m Examples.scripts.run_canonical_astrometry_demo
```

The script builds a Shera three-plane model via ParamSpec/ParameterStore, generates noiseless synthetic binary-star PSFs, and runs a binder/SystemGraph-based gradient descent (with tight priors) to recover astrometric and wavefront parameters using the refactored `InferenceSpec`.

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

- **JAX and immutability**

  dLux optical models are built on JAX and follow a functional, immutable design.
  This means that **model objects cannot be modified in place** — attempting something like:  
  `model.parameter = new_value`  
  will raise an error because the underlying data
  structures (JAX pytrees) are immutable.

  Instead, every update must create a *new* model with the change applied:

    ```python
      model = model.set("parameter_name", new_value)
    ```

    Or for multiple parameters:
    
    ```python
      model = model.set(["param1", "param2"], [val1, val2])
    ```

    This functional update pattern preserves JAX compatibility (JIT, vmap, grad) and
    ensures the entire optical system remains differentiable.

- **`.model()` forward pass**

  This is the standard way to evaluate a dLux optical model. For SHERA this computes a (typically polychromatic) PSF image from the current set of parameters, handling any internal wavelength sampling and normalisation.

- **Eigenmode re-parameterization**

  For inference we diagonalise the Fisher Information Matrix (FIM) and express parameters in the eigenbasis of that matrix. This separates well-constrained from poorly constrained parameter combinations and can improve optimisation:
  - `use_eigen` – toggle between native parameters and eigenmode coefficients.  
  - `whiten_basis` – optionally scale modes by `1/√λ` so all directions have unit variance.  
  - `truncate_k` – keep only the top `k` best-constrained modes.  
  - `truncate_by_eigval` – alternatively, keep all modes with eigenvalue above a chosen threshold.

- **Fresnel propagation and the 3-Plane SHERA model**  
  SHERA uses a custom three-plane optical system to capture beam walk effects, and mirror-specific aberrations that cannot be modeled with a single Fraunhofer propagation.  
  The backend workflow is:

  1. **Primary mirror plane**  
     The pupil field is constructed by combining the primary mirror aperture, wavefront error (WFE), and the diffractive-pupil phase OPD.

  2. **Forward Fresnel propagation to the secondary**  
     The pupil field is propagated to the secondary mirror using a Fresnel Angular Spectrum operator. This produces the near-field amplitude and phase distribution on the secondary.

  3. **Secondary mirror WFE application**  
     Additional WFE (representing alignment errors or surface figure on M2) is applied directly to the propagated field at the secondary plane.

  4. **Backward Fresnel propagation to the primary**  
     The field is then Fresnel-back-propagated to the primary plane, capturing
     how secondary-mirror errors couple back into the entrance pupil.

  5. **Matrix Fourier Transform (MFT) to the focal plane**  
     Finally, we use an MFT to compute the focal-plane field at the desired detector sampling and field of view.
     This produces the monochromatic PSF, which is then internally vectorized and summed over wavelength to yield the polychromatic image.

  This sequence allows the SHERA model to capture the key physical effects
  (M1/M2 misalignments, beam walk, and diffractive pupil structure) needed for
  micro-arcsecond astrometry.


- **Differentiable inference loop**  
  Because everything is built on JAX, we can obtain exact gradients of the loss with respect to all parameters, build the FIM, and run gradient-based optimisers or HMC samplers directly on the optical model.

<!--
## Troubleshooting
- **Zero gradients from “perfect” init:** initialize with *non-zero* tiny perturbations.
- **Slow JIT on first call:** expected; subsequent runs are fast.
- **Diffs not showing?** `git diff -- path/to/file.py` or `git diff --staged`.
-->

## Documentation
(MkDocs configuration pending; this command will serve once docs are added.)
