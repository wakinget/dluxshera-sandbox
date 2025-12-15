# dLuxShera modeling overview

dLuxShera provides a Fresnel-based optical model and inference stack for Shera/TOLIMAN-style astrometric imaging. The primary use case is recovering close-binary parameters (and associated instrument terms) from diffraction-limited images using differentiable optics and gradient-based optimisation. The current stack keeps parameter definitions, derived quantities, and execution order explicit so that optical modeling and inference stay transparent.

## Pipeline at a glance
- **Configuration ➜ forward ParamSpec:** start from a configuration object (e.g., Shera three-plane defaults) and build a forward-facing `ParamSpec` that defines primitives and derived fields.
- **Forward ParameterStore:** instantiate a primitives-only `ParameterStore`, then `refresh_derived` to populate derived quantities via pure transforms.
- **Binder + SystemGraph:** wrap the optics in a `Binder` that owns a `SystemGraph` DAG, exposing a clean "give me a parameter delta ➜ I will produce PSFs/images" interface.
- **Image synthesis:** evaluate the binder to generate polychromatic PSFs or detector images.
- **Losses and optimisation:** construct image NLL/loss functions that pack/unpack θ-vectors to/from stores, and run optimisation loops in θ-space or in eigenmode space via `EigenThetaMap`.
- **Outputs:** inspect recovered parameters, images, and diagnostics via the plotting utilities.

## Choose your path
- **Canonical three-plane demo:** Start with `examples/scripts/run_canonical_astrometry_demo.py` and the walkthrough in [docs/tutorials/canonical_astrometry_demo.md](tutorials/canonical_astrometry_demo.md). This exercises the full V1.0 stack on a Shera-like three-plane system.
- **Canonical two-plane demo (coming soon):** A lighter-weight two-plane variant will mirror the same flow on a simpler optical path while keeping the Binder/SystemGraph, loss, and optimisation pieces identical.
- **Dig deeper:** See the architecture notes in [docs/architecture/](architecture/) for details on parameter specs/stores, binder/system graph execution, inference helpers, and eigenmodes.
