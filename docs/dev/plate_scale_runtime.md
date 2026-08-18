# Runtime plate scale semantics

The forward model declares `optics.plate_scale_as_per_pix` as a **derived**
quantity (from `optics.focal_length_m` and `detector.pixel_pitch_m`). Inference
flows are free to treat the same key as a primitive knob. When running
inference, the model must respect the current store value of
`optics.plate_scale_as_per_pix` so that perturbing θ updates the PSF, loss,
and gradients.

The failure mode to watch for: if the plate scale is cached as a structural
quantity or recomputed unconditionally during evaluation, then the FIM diagonal
for plate scale collapses to zero. Runtime evaluation should always use the
store value when present, even if the forward spec derives it in other
contexts.

## Runtime bindings for cached optics

Optics builders use structural caching keyed on configuration so repeated
evaluations can reuse the same optics geometry. That means any parameters that
are baked into the cached optics object (e.g., Zernike coefficients or plate
scale) must be reapplied from the current `ParameterStore` after cache lookup.

We handle this with *runtime bindings*: explicit `(store_key, optics.set_path)`
tables that describe which store values should be written back into the cached
optics. Only include parameters that live inside the cached optics object.
Do **not** include source, detector, or noise parameters; those are consumed
elsewhere and are not part of the cached optics state.

This avoids “optimizable but no-op” parameters (such as plate scale) where the
inference spec treats a value as a primitive but the optics never updates
post-cache, resulting in zero gradients/FIM entries.
