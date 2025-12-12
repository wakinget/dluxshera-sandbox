# Binder and SystemGraph

## Binders
The Binder is the primary model object in dLuxShera. It combines a configuration, the relevant `ParamSpec`, and a baseline `ParameterStore`, then exposes user-facing methods to evaluate the optical system. Callers supply parameter deltas or θ-vectors, and the Binder handles packing/unpacking, derived parameter resolution, and interaction with the underlying optics. When you “build a Shera model” in the canonical demos, you are creating a Binder and invoking it to produce PSFs or images.

## SystemGraphs
Under the hood, a `SystemGraph` represents the computation as a directed acyclic graph. Nodes correspond to optical elements, intermediate wavefronts or images, and detector steps; edges capture data flow between them. The Binder manages this graph so that typical users do not need to manipulate it directly, but the structure is available for advanced workflows such as inspecting intermediate values or enabling caching.

## How they fit together
ParamSpecs and ParameterStores define what can change. The Binder is the object you call to run the model, taking care of parameter bookkeeping and model evaluation. The SystemGraph is the wiring the Binder orchestrates to generate outputs. Canonical demos interact with the Binder interface and treat the SystemGraph as an implementation detail unless deeper introspection is needed.
