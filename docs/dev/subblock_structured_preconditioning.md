# Structured Preconditioning For Observation Sub-Block Inference

This note explains the structured Fisher/Hessian preconditioning path used by
`examples/recipes/observation_subblock_inference.py`.

The short version:

- A dense packed-theta Fisher matrix is useful for small debugging runs, but it
  scales poorly with the number of frames.
- The current temporal model is `frame_model.kind: independent`, so each frame's
  data term depends only on that frame's active parameters plus any shared active
  parameters.
- With no shared active parameters, the global curvature is exactly block
  diagonal across frames.
- With a small shared block, the global curvature has an arrowhead form that is
  naturally suited to a Schur-complement treatment.

## Why Dense Full-Theta Curvature Does Not Scale

The legacy preconditioner builds

```text
F(theta_ref) = Hessian(loss(theta), theta_ref)
```

over the entire packed active state. If there are `N` frames and `p`
frame-varying active parameters per frame, the frame-only packed dimension is

```text
D = N p
```

A dense Hessian has `D x D` entries. The storage alone grows like `O(D^2)`, but
the more important cost is the automatic differentiation work needed to produce
the dense matrix. In the registration-only case with 20 frames and 3 active
parameters per frame, `D = 60`. That sounds small as a matrix, but a dense
second derivative through the full image-rendering objective asks JAX to trace a
large global computation. For larger frame counts, this quickly becomes the
dominant cost and can kill the process before optimization starts.

The dense path is still valuable for tiny problems because it is simple,
debuggable, and can expose cross-block mistakes. It should be treated as a
debug fallback, not the scalable default for independent-frame sub-blocks.

## Where The Block Structure Comes From

The active state is partitioned as:

```text
theta = [x_0, x_1, ..., x_{N-1}, s]
```

where:

- `x_i` is the frame-varying active vector for frame `i`
- `s` is the shared active vector for the whole sub-block

For the current independent temporal model, the data term is a sum or mean of
per-frame losses:

```text
L(theta) = reduce_i ell_i(x_i, s) + prior(theta) + temporal(theta)
```

Current non-empty priors are not implemented, and the independent temporal term
is zero. That leaves independent per-frame image terms. The important property
is locality:

```text
ell_i does not depend on x_j for j != i
```

This means there is no direct frame-to-frame curvature in the data term.

## Frame-Only Case

When there are no shared active parameters, each per-frame loss is

```text
ell_i(x_i)
```

and the global Hessian/Fisher matrix is:

```text
F =
[ F_0   0    0  ]
[  0   F_1   0  ]
[  0    0   ... ]
```

where:

```text
F_i = Hessian(ell_i, x_i)
```

If `objective.subblock_reduce` is `sum`, each local block enters unchanged. If it
is `mean`, each local block is scaled by `1 / N`.

For the current registration-only run, each `x_i` has:

```text
source.x_position_as
source.y_position_as
source.position_angle_deg
```

so each local block is only `3 x 3`. Computing 20 independent `3 x 3` Hessians
is far cheaper and more stable than asking JAX for one dense `60 x 60` Hessian
through the full sub-block objective.

This block decomposition is exact for the current frame-only independent-frame
data term. The implemented optimizer still consumes a diagonal `lr_vec`, so the
learning-rate vector is built from the diagonal entries of the exact local
blocks. This matches the legacy dense diagonal formula without materializing the
global matrix.

## Frame + Shared Case

When there are shared active parameters, each per-frame loss is:

```text
ell_i(x_i, s)
```

The local Hessian for frame `i` has the block form:

```text
H_i =
[ A_i  B_i ]
[ B_i' C_i ]
```

where:

- `A_i` is frame-local curvature for `x_i`
- `B_i` is frame/shared coupling
- `C_i` is that frame's contribution to shared curvature

The global curvature is not block diagonal anymore. It has an arrowhead form:

```text
F =
[ A_0   0    0   B_0 ]
[  0   A_1   0   B_1 ]
[  0    0   ...  ... ]
[ B_0' B_1' ...  sum_i C_i ]
```

There are still no direct frame-to-frame blocks, but every frame can couple to
the same shared vector. The shared block must be handled globally because the
shared parameters are informed by all frames at once. Treating each frame's
shared curvature independently would double-count the shared variables and miss
their combined uncertainty.

## Schur Complement Intuition

The arrowhead system is attractive because the large part is many independent
small frame blocks, while the global shared block is small. A Newton-like solve
with the full matrix would ask for:

```text
F delta = g
```

In the frame+shared case, this can be viewed as solving many frame equations
plus one shared equation. If the frame blocks `A_i` are invertible, each frame
update can be expressed in terms of the shared update. Substituting those frame
updates into the shared equation produces a much smaller shared system:

```text
S delta_s = rhs_s
S = C - sum_i B_i' A_i^{-1} B_i
```

This `S` is the Schur complement. Practically:

- invert or factor many small `A_i` blocks independently
- accumulate their effect into a small shared system
- solve the shared system once
- recover each frame update independently

For many frames and a few shared parameters, this avoids a dense solve in the
full packed dimension. It also matches the model structure: frames are local,
shared parameters are global, and frame/shared couplings are the only bridge.

## What Is Implemented Now

The implementation lives in:

- `src/dluxshera/inference/structured_preconditioning.py`
- `examples/recipes/observation_subblock_inference.py`

The recipe accepts `experiment.inference.optimizer.preconditioning.method`:

- `auto`
  - uses `frame_block` for independent frames with no shared active parameters
  - uses `frame_shared_structured` for independent frames with shared active
    parameters
- `dense_full_theta`
  - legacy dense packed-theta Hessian/FIM path for small debugging runs
  - aliases: `fim_diag`, `dense`, `dense_fim_diag`
- `frame_block`
  - exact frame-local curvature blocks for independent frames with no shared
    active parameters
  - builds the packed diagonal preconditioning vector from local block diagonals
  - does not materialize the global dense FIM
- `frame_shared_structured`
  - builds local `[frame, shared]` curvature blocks
  - stores frame blocks, shared blocks, and frame/shared coupling blocks
  - accumulates the exact global curvature diagonal for the current `lr_vec`
  - does not yet perform a Schur-complement solve

The current diagonal learning-rate formula remains:

```text
curvature_vec = max(diag(F), curvature_floor)
lr_vec = 1 / (curvature_vec + eps)
```

For `frame_block`, `diag(F)` is assembled from the exact per-frame blocks. For
`frame_shared_structured`, the frame diagonals come from `A_i`, and the shared
diagonal comes from `sum_i C_i`.

Runtime logging reports the selected method, frame count, frame dimension,
shared dimension, and whether a dense global FIM was materialized.

## What Is Left

The main missing piece is an optimizer/preconditioner path that uses the full
structured blocks, not only their diagonal. The natural extension is:

1. Build the same `A_i`, `B_i`, and `C_i` blocks.
2. Stabilize or factor each `A_i`.
3. Accumulate the Schur complement for the shared block.
4. Solve the small shared system.
5. Back-substitute to produce frame updates.
6. Expose this as a structured update rule or a richer preconditioning transform.

That extension should not require changing the block construction API. The
current helper already keeps the frame-local blocks, shared accumulation, and
coupling blocks separate.

## Tradeoffs And Limitations

- The structured methods currently assume `frame_model.kind: independent`.
- Non-empty priors are not implemented in the recipe. When priors are added,
  their curvature must be placed into the correct frame, shared, or coupling
  blocks.
- `frame_block` is exact for the current frame-only independent data term, but
  the consumed optimizer object is still a diagonal preconditioner.
- `frame_shared_structured` builds the arrowhead block data, but the current
  `lr_vec` is the exact diagonal of that arrowhead, not a full Schur-complement
  natural-gradient step.
- Dense full-theta preconditioning remains available for tiny debugging runs and
  for checking structured results on tractable problems.

The guiding rule is: keep curvature in the same structure as the objective. For
independent-frame sub-blocks, frame-local work should stay frame-local, and only
small shared quantities should be accumulated globally.
