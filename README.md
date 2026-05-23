# dLuxShera

High-precision, differentiable astrometric instrument model using dLux (JAX) for auto-diff. This repo contains prescribed 3-plane optical models for the SHERA telescope and inference utilities. This repo allows users to initialize an optical system and set up parameter inference optimization loops 

## TL;DR
- **Overview**: Build and simulate a differentiable three-plane Fresnel optical model (SHERA).
- **Goal**: Recover binary separation, plate scale, and low-order WFE from synthetic images.
- **Status**: Core model + inference utilities are functional; examples and docs in progress.
- **Start here**: See the Quickstart below, then explore the example notebook.

## Documentation

- Conceptual entry point: [docs/tutorials/modeling_overview.md](docs/tutorials/modeling_overview.md)
- Tutorial: [docs/tutorials/canonical_astrometry_demo.md](docs/tutorials/canonical_astrometry_demo.md) alongside `examples/recipes/canonical_astrometry.py` (recipe) and `examples/runners/run_canonical_astrometry.py` (runner)
- Two-plane recipe: `examples/recipes/twoplane_astrometry.py` (read-first variant of the canonical flow)
- Two-plane runner: `examples/runners/run_twoplane_astrometry.py` (execute-first entrypoint)
- [Dev notes](docs/dev/):
  - [Roadmap](docs/dev/roadmap.md) (longer term goals + ideas)
  - [Working Plan](docs/dev/working_plan.md) (near term implementation plan)
  - [Style guide](docs/dev/style_guide.md) (formatting + API conventions)

## Install

This repository supports editable installs via `pyproject.toml`.
Canonical developer installation is:

1. Create a virtual environment.
2. Install editable package + dev extras with `python -m pip install -e ".[dev]"`.
3. Run scripts/tests from that environment (no `PYTHONPATH=src` required).

This will automatically install a temporary **Fresnel-enabled fork of dLux** until the upstream PR is merged.

### Prerequisites (all platforms)

You need:
- **Python 3.11** (recommended)
- **Git** (recommended, but zip downloads also work)
- An editor (we recommend **VS Code**)

If you already have Python installed, skip to “Create the environment”.

---

### 1) Get the repository

**Recommended (Git):**
```bash
cd <folder-where-you-want-projects>
git clone <REPO_URL>
cd <REPO_FOLDER>
```

**Alternative (zip download):**
- Download the repo zip and unzip it somewhere reasonable.
- Then open a terminal in that folder (or `cd` into it).

---

### 2) Create a virtual environment

We create a local `.venv/` folder in the repository root. This keeps dependencies isolated.

**macOS / Linux**
```bash
python3 -m venv .venv
```

**Windows (recommended: use the Python launcher)**
```powershell
py -3.11 -m venv .venv
```

> Why `py -3.11` on Windows?
> It avoids a common Windows issue where `python` points to a Microsoft Store stub instead of your real install.

---

### 3) Activate the environment

**macOS / Linux**
```bash
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt / cmd.exe)**
```bat
.\.venv\Scripts\activate.bat
```

If activation works, your prompt usually shows something like `(.venv)`.

---

### 4) Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Compatibility shims are still available:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

---

### 5) Verify the install

```bash
python -c "import jax, dLux; print('JAX:', jax.__version__); print('dLux:', dLux.__version__)"
```

If versions print without errors, your setup is working.

---

## Quickstart for Non-Python Users

This section is written like “guided troubleshooting”. If something doesn’t work, follow the “If you see X, do Y” notes.

### Windows Quickstart (recommended path)

#### 0) Install the basics

1. **Install Python 3.11 (Windows installer)** from python.org.
   - During install, check:
     - ✅ “Add Python to PATH”
     - ✅ “Install launcher for all users” (recommended)

2. **Install Git** (recommended) so you can `git clone`.

3. **Install VS Code** and the **Python extension**.

#### 1) Make sure Windows is using the right Python

Open **PowerShell** (Start menu → type “PowerShell”).

Run:
```powershell
py -3.11 --version
where python
```

**If `where python` shows something like `...\Microsoft\WindowsApps\python.exe`:**
- That’s a Microsoft Store “app execution alias” stub.
- Fix:
  - Windows Settings → Apps → Advanced app settings → **App execution aliases**
  - Toggle OFF `python.exe` and `python3.exe`
- Then close and re-open PowerShell and re-run `where python`.

#### 2) Open the repo in VS Code

- In VS Code: **File → Open Folder…** and choose the repository root.
- Open a terminal in VS Code: **Terminal → New Terminal**

> If you don’t see a terminal, it’s almost always under the “Terminal” menu.

#### 3) Create + activate the environment

In the VS Code terminal (PowerShell recommended):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**If you see:** “cannot be loaded because running scripts is disabled on this system”
- That means you’re in PowerShell, but script execution is locked down.
- Run this in PowerShell (not cmd.exe):

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

- Close and re-open PowerShell, then activate again:
```powershell
.\.venv\Scripts\Activate.ps1
```

**If `Set-ExecutionPolicy` is “not recognized”**
- You’re not actually in PowerShell (often you’re in cmd.exe or Git Bash).
- Open *Windows PowerShell* explicitly and try again.

#### 4) Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

#### 5) Tell VS Code to use your `.venv`

- Press: `Ctrl+Shift+P`
- Run: **Python: Select Interpreter**
- Choose the interpreter that contains: `.venv\Scripts\python.exe`

This step matters: it ensures VS Code, notebooks, and the “Run Python File” button all use the same environment.

#### 6) Quick verification

```powershell
python -c "import jax, dLux; print('JAX:', jax.__version__); print('dLux:', dLux.__version__)"
```

---

### macOS / Linux Quickstart

1) Install Python 3.11 (system package manager or python.org)

2) Clone and set up:

```bash
git clone <REPO_URL>
cd <REPO_FOLDER>

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python -c "import jax, dLux; print('JAX:', jax.__version__); print('dLux:', dLux.__version__)"
```

3) In VS Code:
- Install the Python extension
- Select interpreter: `.venv/bin/python`

---

### Daily usage (the only two commands you usually need)

**macOS / Linux**
```bash
cd <REPO_FOLDER>
source .venv/bin/activate
code .
```

**Windows (PowerShell)**
```powershell
cd <REPO_FOLDER>
.\.venv\Scripts\Activate.ps1
code .
```

---

### Optional: Using Jupyter notebooks

Launch JupyterLab **from inside the environment**:

```bash
# macOS/Linux
source .venv/bin/activate
jupyter lab
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
jupyter lab
```

Notes:
- To stop Jupyter in the terminal: press `Ctrl+C` and confirm.
- If tqdm progress bars look weird in notebooks, prefer:
  - `from tqdm.auto import tqdm`



## Quickstart (Jupyter Notebook)

Example notebooks are provided under: `examples/notebooks/`

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

## Canonical astrometry demo

The end-to-end astrometry recipe lives in `examples/recipes/canonical_astrometry.py` (read-first) with a thin runner at `examples/runners/run_canonical_astrometry.py` (execute-first).

Run it directly from the repository root:

```bash
python examples/runners/run_canonical_astrometry.py
```

The script builds a Shera three-plane model via ParamSpec/ParameterStore, generates noiseless synthetic binary-star PSFs, and runs a binder/SystemGraph-based gradient descent (with tight priors) to recover astrometric and wavefront parameters using the current `InferenceSpec`.

For the two-plane system, run the mirrored recipe directly (or use the thin runner):

```bash
python examples/runners/run_twoplane_astrometry.py
python examples/recipes/twoplane_astrometry.py
```

---

### Notes

- The new Fresnel propagation utilities for dLux are currently under review. For the time being, dLux installation uses my own local fork. When the PR is fully integrated, these installation instructions will change.
- Notebooks rely on `notebook_setup.py` located in `examples/notebooks/`.
- Packaging is managed via `pyproject.toml` (`pip install -e .` or `pip install -e ".[dev]"`).
- If `python` is unavailable on your host, use `python3` for the same commands.
- `zodiax` is constrained to `<0.5` for current dLux compatibility.

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

This functional update pattern preserves JAX compatibility (JIT, vmap, grad) and ensures the entire optical system remains differentiable.

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
