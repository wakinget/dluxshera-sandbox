# Eigenmodes and EigenThetaMap

## Why eigenmodes?
Some parameter combinations are far better constrained than others. Working in an eigenbasis of a local information metric (such as a Fisher Information Matrix) can highlight these directions, precondition optimisation, and make it easier to interpret which physical effects drive the fit.

## EigenThetaMap
`EigenThetaMap` captures the linear mapping between native θ-space and an eigen-space derived from a curvature estimate. It exposes helpers to map θ → z and z → θ, making whitening or truncating modes straightforward. The map plugs directly into the same Binder-based losses used elsewhere; only the parameter coordinates change.

## Where in the code?
- `dluxshera.inference.optimization.EigenThetaMap` provides the core mapping helpers (`from_fim`, `z_from_theta`, `theta_from_z`).
- `dluxshera.inference.optimization.fim_theta(loss_fn, theta_ref)` computes the local Fisher Information Matrix; `fim_theta_shera(...)` is a convenience wrapper around the same idea for SHERA-specific setups.
- `dluxshera.inference.inference.run_shera_image_gd_eigen(...)` is the turnkey eigen-GD runner built for SHERA image inference.
- `dluxshera.inference.optimization.run_shera_gd(...)` can also run eigen-GD with the right preparation; see `examples/recipes/canonical_astrometry.py` for a complete workflow.

## Observation belief update policy
`dluxshera.inference.observation_belief.update_observation_belief_with_policy`
applies the same eigenmode idea to observation-level Schur summaries without
changing the storage basis. `SubblockSummary` objects remain physical-basis,
label-addressed Schur summaries, and `ObservationBeliefState` remains a physical
parameter-space state. The eigenbasis is only an update-control transform
layered between the full physical posterior update and the applied mean update.

The `physical_full` policy reproduces the legacy `update_observation_belief`
mean update, apart from an optional scalar `update_gain`. `eigen_full` is
intended as an equivalence and validation mode: it projects the full update into
the requested eigenbasis, keeps every mode, and maps back to physical
parameters. `eigen_truncated` and `eigen_damped` are intended for degenerate
M1/M2 slow-state updates where weak information directions should either stay at
the prior/current reference mean or move more cautiously.

When whitening is enabled, the policy builds the basis from the prior-whitened
information matrix using the observation helper for diagonal prior sigmas.
Diagnostics report whether eigenvalues are physical-basis or prior-whitened, the
gate source used for mode retention, per-mode damping factors, retained/rejected
masks, update vectors in both coordinate systems, and dominant physical
contributors.

## Math sketch
Let θ ∈ R^D be the primitive parameter vector used by the optimiser. Choose a reference point θ_ref and evaluate a local Fisher Information Matrix F(θ_ref). With eigen-decomposition

    F = V Λ Vᵀ,   Λ ≥ 0

we have two common coordinate choices.

- **Unwhitened eigen coords:** diagonalise the curvature but keep its scale.

    θ = θ_ref + V z

- **Whitened eigen coords:** rescale so the quadratic form is roughly ‖z‖².

    θ = θ_ref + V Λ^{-1/2} z   (for retained modes)

Truncation keeps the top-K modes (largest eigenvalues) and drops/zeros the rest. This can improve conditioning, reduce optimisation noise, and focus on the directions the data actually constrain.

Practical caveats:
- The metric is local: F depends on θ_ref and can change as the fit moves.
- Eigenvectors are only defined up to a sign flip, so z coordinates may flip sign between runs.
- Near-degenerate eigenvalues can rotate the eigen-basis within their subspace.

## Mapping: θ ↔ z
`EigenThetaMap` provides explicit conversions in both directions:

- θ → z via `z_from_theta` (project a θ vector into eigen coordinates).
- z → θ via `theta_from_z` (reconstruct θ from eigen coordinates).

Whitening conceptually means scaling each retained eigen direction by √λ so that a unit step in z corresponds to an equal-curvature step in θ. When truncating, dropped modes are set to zero in z, and the reconstructed θ stays in the retained subspace around θ_ref.

## Typical workflow
For a worked end-to-end routine, `examples/recipes/canonical_astrometry.py` shows how to build the loss, compute the FIM, and run eigen-GD.

High-level pseudocode for the same flow:

    loss_fn = make_binder_image_nll_fn(...)
    theta_ref = theta0  # or a later iterate
    F = fim_theta(loss_fn, theta_ref)
    eigen_map = EigenThetaMap.from_fim(F, theta_ref, truncate=K, whiten=True)
    z0 = eigen_map.z_from_theta(theta0)
    z_star = run_gd_in_z(loss_fn, z0, eigen_map)
    theta_star = eigen_map.theta_from_z(z_star)

The key idea is that optimisation runs in z while the loss still evaluates in θ, so only the coordinate transform changes.

## Worked example
If you want an executable reference, `examples/recipes/canonical_astrometry.py` runs a full eigen-GD loop and is a good starting point for adapting to your own targets.

## See also
- `docs/architecture/inference_and_loss.md` for binder-first loss construction.
- `docs/architecture/optimization_artifacts_and_plotting.md` for optimisation artifacts and θ-space metadata.
- `examples/notebooks/Shera_Eigen_Inference_Example.ipynb` for a notebook walkthrough.
