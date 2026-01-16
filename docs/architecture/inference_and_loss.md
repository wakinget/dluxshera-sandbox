# Inference helpers and loss construction

## Inference pipeline
The standard flow mirrors the canonical astrometry demos: choose a configuration, build the matching `ParamSpec` and baseline `ParameterStore`, construct a Binder, and use it to generate model images or PSFs. Observed data are compared to these outputs through a loss or negative log-likelihood, and an optimiser updates parameters in θ-space. This pipeline treats the Binder as the execution engine while keeping parameter bookkeeping aligned with the spec and store.

## Loss helpers
Binder-aware loss helpers accept a parameter vector θ, convert it to a `ParameterStore` delta, and call the Binder to produce model images. They then evaluate a simple noise model (for example, Gaussian image NLL) against observed data and optionally add prior penalties for MAP objectives. These helpers are the recommended starting point for new image-based experiments because they preserve the packing/unpacking logic and keep derived parameters refreshed automatically.

## Optimisation
Gradient-based optimisers (such as small Optax loops) consume the Binder-backed losses and apply updates to θ. The helpers in the canonical demos are intentionally lightweight convenience routines rather than a full inference framework; they make it easy to prototype and to plug in alternative parameterisations like eigenmodes when desired. More advanced inference methods (e.g., NumPyro or HMC) can layer on later without changing the Binder-facing loss surface.
