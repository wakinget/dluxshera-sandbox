# Inference helpers and loss construction

## Image NLL / loss functions
- Loss helpers wrap the Binder so callers work in θ-space while the Binder handles stores and optics evaluation.
- Packing/unpacking utilities convert between flat θ vectors and `ParameterStore` deltas, keeping derived fields refreshed as needed.
- Gaussian image NLL helpers expose reduction options and can be combined with prior penalties for MAP objectives.

## Optimisation flows
- Gradient descent loops (e.g., Optax) consume θ-space losses and apply updates to flat parameter vectors.
- The canonical demo uses this flow directly, optionally layering a quadratic prior penalty for regularisation.
- Eigenmode optimisation wraps the same Binder-based loss with an `EigenThetaMap` for alternative coordinates.
