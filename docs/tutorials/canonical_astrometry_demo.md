# Canonical astrometry demo (three-plane)

The canonical recipe in `examples/recipes/canonical_astrometry.py` builds a Shera-like three-plane optical system, generates synthetic binary-star data, and recovers the scene with gradient-based optimisation. A thin runner lives at `examples/runners/run_canonical_astrometry.py`. The recipe highlights the current stack: config resolution (`load_user_config` ➜ `resolve_config`), forward-spec composition, binder-first evaluation, image NLL construction, and optimisation in θ-space (with optional eigen-θ runs).

## What the demo covers
- Shera-style three-plane optical path with Fresnel propagation.
- Declarative detector composition via `system.detector.layers` (defaults to identity/no-op layers when omitted).
- Synthetic truth generation for a binary target and noisy observations.
- Pure θ-space gradient descent for MAP estimation, with an optional eigen-θ parameterisation.
- Plotting helpers for PSF comparison and parameter history.

## Step-by-step walkthrough (matches the recipe)
- **Load + resolve config (user → system/experiment):** The script loads YAML/JSON via `load_user_config`, then calls `resolve_config` to apply presets, deep-merge overrides, and validation. The resolved config exposes `system` (physical model) and `experiment` (workflow settings). Detector layers can be overridden by editing `system.detector.layers` before composing the spec.
- **Compose the forward spec (system → spec):** `compose_forward_spec(system_cfg)` builds the contract from source/optics/detector builders. All parameter keys are component-prefixed (e.g., `source.*`, `optics.*`, `detector.*`), and bindings capture runtime paths for binder access and runtime patching.
- **Create the base store (spec → values):** `ParameterStore.from_spec_defaults(forward_spec)` seeds primitives, then `.refresh_derived(forward_spec)` fills derived values such as `optics.plate_scale_as_per_pix`. The same store serves as both truth and inference baseline.
- **Build the Binder (system + spec + store):** `SheraBinder(system_cfg, forward_spec, forward_store)` is the runtime model. `binder.model()` uses cached optics/detector; pass `binder.strip_structural(delta_store)` for per-call, non-structural updates. Structural edits still require `binder.update_store(..., allow_rebuild=True)`.
- **Simulate observations (Binder → data):** Evaluate the binder for a noiseless PSF, then add optional noise (Gaussian or Poisson) to form the observed image. Variance is set from the noiseless PSF with a floor to avoid divide-by-zero in the NLL.
- **Define inference layout (spec → subset):** Choose inference keys from the forward spec and build `inference_subspec = forward_spec.subset(infer_keys)`. Packing/unpacking (`pack_params`, `store_unpack_params`) operate on this subspec; derived-labelled keys remain inferable (store-wins) without special casing.
- **Initialise and run optimisation (θ → best-fit):** Priors are defined per key; initial stores come from prior samples or truth depending on `experiment.init.mode`. Loss construction uses `make_binder_nll_fn`, with optional eigenmode wrapping via `EigenThetaMap`. The optimisation loop updates θ in pure θ-space while binder calls stay store-based.
- **Plot + inspect results (artifacts + figures):** The script writes PSF comparisons, parameter histories, and optional eigen spectra to the run directory alongside optimisation artifacts. See `docs/architecture/optimization_artifacts_and_plotting.md` for layout details.

## Running the script
From the repository root:

```bash
python examples/runners/run_canonical_astrometry.py
```

Use the `--fast` runner flag for a quick smoke run, or edit the `FAST_MODE`/`SAVE_PLOTS` toggles near the top of the recipe:

```bash
python examples/runners/run_canonical_astrometry.py --fast
```
