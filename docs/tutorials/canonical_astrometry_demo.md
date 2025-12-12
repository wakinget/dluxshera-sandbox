# Canonical astrometry demo (three-plane)

The canonical demo in `Examples/scripts/run_canonical_astrometry_demo.py` builds a Shera-like three-plane optical system, generates synthetic binary-star data, and recovers the scene with gradient-based optimisation. The script highlights the current stack: `ParamSpec`/`ParameterStore`, `DerivedResolver`, `Binder`/`SystemGraph`, image NLL construction, and optimisation in θ-space (with optional eigen-θ runs when enabled).

## What the demo covers
- Shera-style three-plane optical path with Fresnel propagation.
- Synthetic truth generation for a binary target and noisy observations.
- Pure θ-space gradient descent for MAP estimation, with a note about eigen-θ optimisation when enabled in the script.
- Plotting helpers for PSF comparison and parameter history.

## Step-by-step walkthrough
- **Build the config and forward ParamSpec:** The script seeds a Shera configuration, then calls the forward `ParamSpec` builder (see `build_forward_spec` in the script) to define primitives and derived fields for inference.
- **Create the base forward ParameterStore:** Use the spec defaults to create a primitives-only store and call `refresh_derived` to populate values such as pixel scale and log flux via `DerivedResolver` transforms.
- **Construct the Binder/SystemGraph:** Instantiate a `SheraThreePlaneBinder` (optionally `use_system_graph=True`) so evaluation is a single `binder.model(store_delta)` call that runs through the DAG.
- **Generate synthetic data:** Draw a "truth" `ParameterStore`, evaluate the binder to get a noiseless image, and add Gaussian noise to obtain observations.
- **Build the binder-based loss:** `make_binder_image_nll_fn` returns a θ-packing loss and the initial θ vector. The demo adds a quadratic prior penalty for MAP optimisation.
- **Run gradient descent:** The main loop applies Optax updates to θ in pure θ-space; when eigenmode helpers are enabled, the same loss is wrapped via `EigenThetaMap` for optimisation in eigen-θ coordinates.
- **Inspect results:** The script saves PSF comparison plots and parameter history grids (see `plot_psf_comparison` and `plot_parameter_history_grid`), writing outputs when an output directory is provided.

## Looking ahead: two-plane canonical demo
A forthcoming two-plane canonical demo will follow the same structure with a simplified optical path. Expect the same flow—config ➜ forward spec ➜ base store ➜ Binder/SystemGraph ➜ loss ➜ optimisation—but with fewer planes to help new users get started quickly.

## Running the script
From the repository root:

```bash
python -m Examples.scripts.run_canonical_astrometry_demo
```

Use the `fast=True` flag for a quick smoke run or provide an output directory to save figures, for example:

```bash
python -m Examples.scripts.run_canonical_astrometry_demo --fast True --output_dir /tmp/shera_demo
```
