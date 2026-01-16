# dLuxShera modeling overview

dLuxShera provides a Fresnel-based optical model and inference stack for Shera/TOLIMAN-style astrometric imaging. The primary use case is recovering close-binary parameters (and associated instrument terms) from diffraction-limited images using differentiable optics and gradient-based optimisation. The current stack keeps parameter definitions, derived quantities, and execution order explicit so that optical modeling and inference stay transparent.

## Pipeline at a glance
- **Configuration ➜ forward ParamSpec:** start from a configuration object (e.g., Shera three-plane defaults) and build a forward-facing `ParamSpec` that defines primitives and derived fields.
- **Forward ParameterStore:** instantiate a primitives-only `ParameterStore`, then `refresh_derived` to populate derived quantities via pure transforms.
- **Binder-only evaluation:** wrap the optics in a `Binder`, exposing a clean "give me a parameter delta ➜ I will produce PSFs/images" interface.
- **Update/delta workflow:** use `binder.model(store_delta)` for per-call overlays; if you truly need a new baseline (or a structural change), create a new binder via `binder.update_store(...)`.
- **Image synthesis:** evaluate the binder to generate polychromatic PSFs or detector images.
- **Losses and optimisation:** construct image NLL/loss functions that pack/unpack θ-vectors to/from stores, and run optimisation loops in θ-space or in eigenmode space via `EigenThetaMap`.
- **Outputs:** inspect recovered parameters, images, and diagnostics via the plotting utilities.

## Choose your path
- **Canonical three-plane demo:** Start with the read-first recipe at `examples/recipes/canonical_astrometry.py` and the walkthrough in [docs/tutorials/canonical_astrometry_demo.md](canonical_astrometry_demo.md). The execute-first runner lives at `examples/runners/run_canonical_astrometry.py`. This exercises the full V1.0 stack on a Shera-like three-plane system.
- **Canonical two-plane demo:** The read-first recipe at `examples/recipes/twoplane_astrometry.py` mirrors the three-plane workflow on the two-plane Shera system, using the same binder, loss, and eigenmode optionality.
- **Dig deeper:** See the architecture notes in [docs/architecture/](architecture/) for details on parameter specs/stores, binder-based execution, inference helpers, and eigenmodes.
