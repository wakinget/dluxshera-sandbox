# Runtime plate scale semantics

The forward model may declare `system.plate_scale_as_per_pix` as a derived
quantity for truth/data generation, but inference specs are allowed to treat
that same key as a primitive knob. When running inference, the model must
respect the current store value of `system.plate_scale_as_per_pix` so that
perturbing θ updates the PSF, loss, and gradients.

The failure mode to watch for: if the plate scale is cached as a structural
quantity or recomputed unconditionally during evaluation, then the FIM diagonal
for plate scale collapses to zero. Runtime evaluation should always use the
store value when present, even if the forward spec derives it in other
contexts.
