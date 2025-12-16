# Canonical astrometry demo (three-plane)

The canonical demo in `examples/scripts/run_canonical_astrometry_demo.py` (implemented in `dluxshera.demos.canonical_astrometry`) builds a Shera-like three-plane optical system, generates synthetic binary-star data, and recovers the scene with gradient-based optimisation. The script highlights the current stack: `ParamSpec`/`ParameterStore`, `DerivedResolver`, `Binder`/`SystemGraph`, image NLL construction, and optimisation in θ-space (with optional eigen-θ runs when enabled).

## What the demo covers
- Shera-style three-plane optical path with Fresnel propagation.
- Synthetic truth generation for a binary target and noisy observations.
- Pure θ-space gradient descent for MAP estimation, with a note about eigen-θ optimisation when enabled in the script.
- Plotting helpers for PSF comparison and parameter history.

## Step-by-step walkthrough
- **Build the config and forward ParamSpec:** The script seeds a Shera configuration, then calls the forward `ParamSpec` builder (see `build_forward_spec` in the script) to define primitives and derived fields for inference. Shera configs are frozen dataclasses for structural hashing/caching, so tweak predefined designs (e.g., `SHERA_TESTBED_CONFIG`) with the ergonomic `.replace(...)` helper rather than attribute assignment, for example:

  ```python
  cfg = SHERA_TESTBED_CONFIG.replace(
      primary_noll_indices=(4, 5, 6, 7, 8),
      secondary_noll_indices=(4, 5, 6, 7, 8),
      oversample=4,
  )
  ```
- **Create the base forward ParameterStore:** Use the spec defaults to create a primitives-only store and call `store.refresh_derived(forward_spec)` to populate values such as pixel scale and log flux. Derived transform modules are registered lazily, so the store method is the one-stop "compute deriveds" entrypoint. We typically do not construct a separate `inference_store`; instead we reuse `forward_store` (or `init_store`) as the base store for packing/unpacking.
- **Define the inference view:** Choose the subset/order of parameters to infer (e.g., astrometry-only) and build an inference subspec directly from the forward spec using `make_inference_subspec`. Validate that the base store already contains the needed keys (and shapes) with `validate_inference_base_store` before packing θ vectors, for example:

  ```python
  inference_subspec = make_inference_subspec(
      base_spec=forward_spec,
      infer_keys=["binary.separation_as", "binary.position_angle_deg"],
      cfg=cfg,
      include_secondary=False,
  )
  validate_inference_base_store(forward_store, inference_subspec)
  ```
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
python examples/scripts/run_canonical_astrometry_demo.py
```

Use the `fast=True` flag for a quick smoke run or provide an output directory to save figures, for example:

```bash
python examples/scripts/run_canonical_astrometry_demo.py --fast True --save-plots-dir /tmp/shera_demo
```
