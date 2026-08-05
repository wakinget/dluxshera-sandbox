# Observation-Level Estimation Design Note

## Purpose

This note defines a first detailed design for **observation-level estimation**
built on top of the existing observation sub-block workflow.

The current sub-block inference prototype is intentionally narrow: it solves a
joint local problem over a short stack of frames while holding the shared system
state fixed. That is a useful and necessary first milestone. However, the
mission-aligned inference story is larger:

- each sub-block is solved under a current **belief** about the slowly varying
  or shared state,
- the sequence of solved sub-blocks should contribute evidence back to that
  belief,
- some parameters may be estimated directly inside individual sub-block solves,
- and future updates to the belief may require selective or global refresh of
  previously processed sub-blocks.

This document expands that narrative into a more concrete architecture and
introduces first-draft schemas for the two main handoff objects:

- ``SubblockSummary``
- ``ObservationBeliefState``

The goal is not to finalize the full mission algorithm immediately. The goal is
to define a coherent, testable, and computationally plausible direction that can
guide staged implementation.

## Relationship to existing design notes

This note is intended to complement, not replace,
``docs/dev/obs_subblock_inference_design.md``.

That note already defines:

- the distinction between **belief state**, **active inference state**, and
  **frozen state**,
- the idea that sub-block inference is a **joint local block problem**,
- and the long-term hierarchy of:
  - local sub-block inference,
  - belief propagation and updating across sub-blocks.

This note focuses specifically on the **observation-level layer** that sits
above the current local block solves. In particular, it addresses:

- what information should flow upward from a solved sub-block,
- what the observation-level estimator should operate on,
- how to separate frame-local, sub-block-shared, and observation-level state,
- how to represent Schur-complement Fisher summaries,
- how to support different linearization points across blocks,
- how observation-level eigenmodes should be treated,
- how and when previously solved sub-blocks should be revisited,
- and how to validate the idea without immediately simulating a full mission-
  scale observation.

## Scope

In scope:

- observation-level estimation narrative and terminology
- conceptual formulation of block-to-observation information flow
- first-draft schemas for sub-block summaries and observation belief state
- Schur-complement Fisher summaries as the preferred reduced sensitivity object
- parameter allocation between frame-local, sub-block-shared, and
  observation-level state
- observation-level eigenmode parameterization as an optional transform layer
- design guidance for refresh / revisit policy
- staged toy-scale validation strategy

Out of scope:

- a final mission-ready recursive estimator
- a full Bayesian filter implementation
- final data-product formats
- final matrix / tensor storage conventions
- mission-scale runtime benchmarking
- exact thresholds for refresh decisions
- final decisions about which physical parameter must live at which inference
  level

## Motivation

A mission-like observation may contain many thousands of frames and many
hundreds or thousands of sub-blocks.

For example, under a working assumption of:

- frame cadence near 20 Hz,
- approximately 1 second per fixed-ROI sub-block,

then a 30-minute observation would contain roughly:

- 36,000 frames,
- 1,800 sub-blocks.

The current local solve is already tractable at the sub-block scale. The
observation-level challenge is therefore not how to directly optimize one giant
problem over all frames. The challenge is:

> how to carry forward enough scientifically meaningful information from each
> sub-block that the slow/shared state can be updated across the observation
> without explicitly re-solving all frame-varying nuisance variables at every
> step.

That is the central tractability problem this note addresses.

## High-level narrative

The intended observation-level narrative is:

1. A current belief state defines the assumed slow/shared state for a set of
   sub-block solves.
2. Each sub-block is solved locally under that assumed state, with a
   config-defined active state.
3. The sub-block emits a summary that captures:
   - the local optimum,
   - fit diagnostics,
   - the state partition used for that block,
   - and local sensitivity / curvature information relevant to the
     observation-level state.
4. Multiple such summaries are combined at the observation level to update the
   slow/shared belief state.
5. After the belief changes, previously solved sub-blocks are not automatically
   re-solved.
6. Instead, previously solved sub-blocks are revisited only when diagnostics or
   validity tests indicate that their cached local approximation is no longer
   trustworthy.

This is the core cascade:

```text
sub-block solve
  -> summary handoff
  -> observation-level update
  -> selective or global refresh as needed
```

## Retrospective information-rate diagnostics

Preserved one-second Schur summaries can also be audited after a campaign to
measure how quickly each coupled slow-state mode accumulates information.  This
is an information-only diagnostic: it uses the reduced information matrices
from accepted summaries, not their science scores, and it does not change
campaign execution or historical reference updates.

For accepted sub-block summaries with reduced slow-state information `S_b` and
audited duration `dt_b`, the cumulative prefix information is:

```text
S_prefix(n) = sum_{b<=n} S_b
T_prefix(n) = sum_{b<=n} dt_b
S_rate(n) = S_prefix(n) / T_prefix(n)
```

The reviewer reconstructs the initial observation prior covariance `C0` and
forms its symmetric square root `W`, with `W @ W.T = C0`.  Prior-whitened
information gain is:

```text
G_prefix = W.T @ S_prefix @ W
G_rate = W.T @ S_rate @ W
```

Eigenvalues of `G_rate` are prior-relative information-gain rates in `1/s`.
For a mode rate `r_k`, the linear stationary approximation is
`gamma_k(T) ~= r_k * T`, with variance contraction
`variance_post / variance_prior ~= 1 / (1 + gamma_k)`.  The diagnostic
information replacement time is `tau_info_k = 1 / r_k` for positive rates.

Because prefix eigenvectors can rotate, swap, or change sign, the audit uses a
single canonical reference spectrum.  The canonical spectrum is the
prior-whitened information-rate matrix pooled over the final configured tail
windows.  Its eigenvectors are ordered by decreasing gain rate and signed
deterministically by making the largest-absolute loading positive.  Threshold
crossings, adaptive-cadence candidates, and drift scenarios use fixed canonical
projected gains:

```text
projected_gain_k(prefix) = v_k.T @ G_prefix @ v_k
```

Instantaneous prefix and per-window eigenspectra are aligned back to the
canonical basis by maximum absolute overlap.  Nearly degenerate canonical
eigenvalues are treated as subspaces; individual loadings inside those groups
are marked as non-unique, while principal-angle/subspace-overlap diagnostics are
authoritative.

The same audit reports physical composition by grouping whitened squared
loadings into source parameters, plate scale, M1 Zernikes, M2 Zernikes, and
other labels.  It also reports coordinate-marginal covariance contraction from
`prior_precision + accumulated_information`; those marginals are coupled
covariance diagnostics, not independent physical-parameter information rates.

Schur-reduced Fisher matrices can contain tiny negative eigenvalues from
finite-precision linear algebra.  Information-rate ingestion preserves raw
eigenvalue diagnostics, then applies the shared PSD tolerance policy
`PSD_ATOL + PSD_RTOL * max(max(abs(raw_eigenvalues)), 1.0)`.  If a negative
eigenvalue is below that tolerance in magnitude, it is treated as numerical
roundoff and explicitly projected to zero by eigendecomposition/reconstruction
before any downstream information calculation.  Materially indefinite matrices
remain errors and are not silently regularized.  This projection is a numerical
consistency correction, not a physical regularization assumption; all canonical
spectra, prefixes, cadence diagnostics, sequential precision updates, and
final-information invariance checks use the PSD-projected accepted matrices.

Adaptive-cadence tables from this audit evaluate only whether information is
sufficient to support an update for the top canonical modes.  A future
controller still needs an innovation or requested-update criterion.  In other
words, information says whether an update is supportable; score/innovation says
whether a meaningful update is requested.

The audit also includes a sequential information-only gating diagnostic.  This
is a covariance-only cadence simulation over frozen Schur factors, not a
recovered adaptive estimator trajectory.  The canonical late-tail
prior-whitened eigenvectors `v_k` are converted once into physical directions

```text
d_k = W0 @ v_k
```

and normalized so that `d_k.T @ P0 @ d_k = 1`, where `P0 = inverse(C0)`.  These
fixed physical directions preserve the interpreted late-tail mode identities
even as the covariance contracts.  At a later precision state `P_current`, the
buffered information gain for a fixed direction is evaluated as

```text
gamma_k = (d_k.T @ S_buffer @ d_k) / (d_k.T @ P_current @ d_k)
```

which is equivalent to normalizing `d_k` in the current precision metric before
projection.  When a block closes, precision is updated additively:

```text
P_after = P_before + S_buffer
```

and then symmetrized and checked for positive definiteness.  No mean, score,
innovation, requested update, or reference trajectory is updated.  Because
`d_k.T @ P_current @ d_k` increases as information accumulates, identical
absolute subblock information produces smaller current-relative gain later in
the sequence; acquisition blocks can therefore lengthen naturally or reach a
configured maximum latency.

Two sequential scopes are reported.  In `window_restart`, precision resets to
`P0` at each historical 30-second window and the open buffer is flushed at the
window end.  This compares local information geometry between historical
references.  In `observation_carry_window_bounded`, precision carries across
the whole observation, but open buffers are still flushed at historical window
boundaries so one temporary acquisition block does not mix summaries generated
on opposite sides of a historical reference update.  Every accepted summary is
retained, and each policy reports final-information invariance against
`P0 + sum(S_b)` to verify that block boundaries do not change the final
precision.

Sequential policies use named canonical mode sets resolved from squared
whitened physical loadings.  `astrometric_core` uniquely assigns separation and
plate scale; `source_core` adds log flux and contrast; `high_information_calibration`
combines the astrometric core with the strongest WFE-dominated modes; and
`all_trackable_initial` includes modes whose late-tail rate times the maximum
acquisition duration exceeds the requested gain threshold.  Assignment is a
deterministic maximum-weight unique matching with weak/ambiguous loading
cautions.

Strict degeneracy remains defined by `DEGENERACY_RTOL = 1e-3`.  A separate
quasi-degeneracy tolerance, defaulting to `1e-2`, marks near-equal modes where
individual eigenvectors may rotate but the subspace can remain stable.  Quasi
groups are not relabeled as formal degeneracies; the preferred diagnostic is
subspace singular value or principal angle across windows, with individual
loadings retained but marked with a caution.

The sequential products are useful for information accumulation, covariance
contraction, information-normalized cadence comparisons, and conservative
schedule diagnostics.  They cannot simulate a posterior score, posterior mean,
innovation gate, hypothetical relinearized likelihood factor, requested state
update, or nonlinear reference trajectory.  Information gating is therefore only
one half of a future controller; innovation/requested-update gating must be
specified separately.

Drift-observability scenarios are illustrative.  For one whitened canonical
mode with measurement rate `r` and random-walk process variance rate `q`, the
scalar diagnostic model is `dP/dt = q - r * P**2`.  The steady-state variance is
`sqrt(q / r)`, the steady-state standard deviation is `(q / r)**0.25`, and the
maximum process variance rate for target steady-state sigma fraction `f` is
`q_max = r * f**4`.  These values inform stability discussions but are not
formal instrument requirements.

This diagnostic supports design of a future `acquire_then_accumulate`
controller: predicted initialization, adaptive reference acquisition, then
fixed-reference precision accumulation.  It does not implement that controller
or simulate a prospective adaptive trajectory from frozen historical summaries.

For the first demonstration, the preferred operational pattern is a **batch
observation update**:

```text
solve many sub-blocks independently
  -> collect many SubblockSummary objects
  -> apply one observation-level belief update
  -> emit one ObservationBeliefState for that observation
```

The same summary contract should also be compatible with a later sequential
filtering interpretation, but the first demo does not need to update the belief
after every one-second sub-block.

## Core principle

The observation-level problem should not be framed as:

- “fit all frame-varying variables again across the whole observation,”

nor as:

- “ignore the image likelihood and operate only on a table of recovered local
  state estimates.”

Instead, it should be framed as a **reduced estimation problem** in which each
sub-block contributes a locally valid approximation of how its image-domain
likelihood depends on the slow/shared state after the local nuisance structure
has been handled.

This leads naturally to a division of labor:

- the **sub-block layer** handles fast nuisance structure and local image fit,
- the **observation layer** handles low-dimensional slow/shared inference and
  consistency across blocks.

## Exact versus practical formulation

### Exact view

Let:

- ``Theta`` denote the observation-level slow/shared variables of interest,
- ``psi_b`` denote block-shared variables that may be actively estimated inside
  sub-block ``b`` but not necessarily propagated as observation-level state,
- ``phi_b`` denote the local fast nuisance variables for block ``b``.

Then an exact observation-level objective would conceptually resemble:

```text
L_obs(Theta) = sum_b min_{psi_b, phi_b} L_b(phi_b, psi_b, Theta)
```

or, in a marginalized form:

```text
L_obs(Theta) = -sum_b log ∫∫ exp(-L_b(phi_b, psi_b, Theta)) dphi_b dpsi_b
```

This exact view is useful conceptually because it makes the coupling explicit:

- if ``Theta`` changes,
- the locally optimal ``phi_b`` and ``psi_b`` may also change.

### Practical view

The practical design goal is not to repeatedly solve the exact problem from
scratch. The practical goal is:

- to summarize each block well enough that **small** changes in ``Theta`` can
  be handled using cached local information,
- to represent any block-local shared estimates explicitly,
- and to allow **selective or global refresh** when cached summaries are no
  longer accurate enough.

The observation-level problem is therefore an approximation problem as much as
an estimation problem.

## State allocation across inference levels

A key design requirement is that the same physical parameter may reasonably move
between inference levels as the algorithm matures. The implementation should
therefore avoid baking in one permanent classification.

### Frame-local fast state

Frame-local fast state varies from frame to frame and is usually eliminated at
or below the sub-block boundary.

Current first-demo examples:

- ``source.x_position_as``
- ``source.y_position_as``
- ``source.position_angle_deg``

These terms are active in the registration-only sub-block solve. Their recovered
values are useful diagnostics, but they are not the primary observation-level
state.

### Sub-block-shared local state

Sub-block-shared local state is constant within one sub-block but may be fit
inside that sub-block as an active shared parameter.

Possible examples:

- ``optics.plate_scale_as_per_pix``
- ``source.log_flux_total``
- selected low-dimensional calibration terms

If a parameter is promoted to the sub-block shared state, the observation-level
update may treat it in one of several ways:

1. **Eliminate it locally** with the nuisance state and leave it out of
   ``Theta``.
2. **Aggregate it descriptively** across blocks, for example reporting a mean,
   scatter, or drift diagnostic without using it as a formal belief-state
   dimension.
3. **Promote it into ``Theta``** when the observation-level layer should estimate
   one coherent value or slow drift across many blocks.

For example, if ``optics.plate_scale_as_per_pix`` is estimated directly inside
each sub-block, then the main observation-level belief update may choose to
exclude plate scale from ``Theta`` and instead report blockwise plate-scale
statistics as an auxiliary product.

### Observation-level slow state

Observation-level slow state contains parameters whose values should be
estimated from many sub-blocks together.

The design target is the **full canonical slow-state family**, with parameter
groups enabled or disabled by configuration. Candidate groups include:

- binary/source terms:
  - ``source.separation_as``
  - ``source.contrast``
  - ``source.log_flux_total``
  - binary position-angle-like terms, subject to key semantics cleanup
- optical scale terms:
  - ``optics.plate_scale_as_per_pix``
- low-order aberration terms:
  - selected ``optics.primary.zernike_coeffs_nm[...]`` components
  - selected ``optics.secondary.zernike_coeffs_nm[...]`` components

The first implementation should support the full layout contract from the
beginning, even if early runs disable some groups.

### Important naming / semantics note

The current registration-only solve uses ``source.position_angle_deg`` as a
frame-varying registration-like term. The canonical astrometry recipes may also
use source/binary position angle as a science parameter. Before the full
canonical observation-level state is finalized, the implementation should audit
whether these are truly the same physical parameter or whether the store keys
need a clearer separation between:

- binary/source position angle,
- frame registration roll / orientation,
- and any detector or ROI orientation term.

Until that is clarified, the observation-level ``ThetaLayout`` should avoid
simultaneously treating the same key as both a frame-local nuisance variable and
an observation-level slow parameter.

## Why state summaries are needed

A plain recovered per-frame registration table is not enough for the
observation-level layer.

It is useful as an output artifact and for debugging, but by itself it does not
capture:

- how sensitive the block fit is to candidate slow/shared parameters,
- how much of the block fit is absorbed by fast nuisance variables,
- whether a later change in the slow/shared belief is small enough for a local
  approximation to remain valid,
- or how strongly this block should influence a later observation-level update.

This motivates a richer local handoff object: ``SubblockSummary``.

## Fast-state elimination and the Schur-complement viewpoint

A central idea in the observation-level design is that local nuisance state
should be handled locally and then **eliminated**, approximated, or profiled out
when passing information upward.

For the first demo, the desired reduced representation is the
**Schur-complement Fisher summary**.

Suppose a local block objective has curvature partitioned into:

- observation-level variables ``Theta``,
- local nuisance variables ``zeta_b``.

Here ``zeta_b`` may include only frame registration variables:

```text
zeta_b = phi_b
```

or it may include both frame registration and sub-block-shared active variables
that are not part of the observation-level belief update:

```text
zeta_b = [psi_b, phi_b]
```

Then the local curvature can be partitioned as:

```text
H_b = [[H_ThetaTheta, H_Thetazeta],
       [H_zetaTheta,  H_zetazeta ]]
```

The effective local information on the observation-level variables after
eliminating local nuisance terms is governed by the Schur complement:

```text
S_b = H_ThetaTheta - H_Thetazeta H_zetazeta^{-1} H_zetaTheta
```

This matters for two reasons:

1. **Identifiability**
   - It quantifies how much information remains on candidate observation-level
     parameters after allowing the local registration and block-shared nuisance
     terms to move.

2. **Observation-level handoff**
   - It provides a natural reduced representation of what the block contributes
     to the observation-level slow/shared estimation problem.

The first implementation should target this Schur-reduced Fisher object as the
primary sensitivity summary. It may use damping, pseudo-inverses, eigenvalue
floors, or rank diagnostics when ``H_zetazeta`` is singular or poorly
conditioned.

## Local quadratic summaries and linearization points

Each ``SubblockSummary`` should represent a local quadratic surrogate for the
block objective with respect to the observation-level state.

For block ``b``:

```text
L_b(Theta) ≈ c_b
           + g_b^T (Theta - Theta_ref_b)
           + 1/2 (Theta - Theta_ref_b)^T S_b (Theta - Theta_ref_b)
```

where:

- ``Theta_ref_b`` is the physical-basis linearization point used for the summary,
- ``g_b`` is the Schur-reduced gradient / score term with respect to ``Theta`` at
  ``Theta_ref_b``,
- ``S_b`` is the Schur-reduced Fisher / curvature matrix,
- and ``c_b`` is an optional constant that is usually not needed for the update.

### Common linearization point

For the first demo, all sub-block summaries may be linearized around the same
``Theta_ref``. This is the easiest case:

```text
Theta_ref_0 = Theta_ref_1 = ... = Theta_ref
```

The observation update can then accumulate curvature and score terms directly
around the same reference point.

### Different linearization points

The design should not assume this will always be true. More complex workflows
may create summaries around different references. This can happen when:

- sub-blocks are processed after belief-state updates have already occurred,
- a subset of blocks is selectively refreshed under a newer belief,
- a block-level shared parameter is estimated locally and changes the effective
  expansion point,
- or different observations are combined after being processed under different
  priors.

For that reason, each summary must explicitly carry its own
``Theta_ref_b`` and enough information to convert its local quadratic into a
common information form.

Starting from:

```text
L_b(Theta) ≈ c_b
           + g_b^T (Theta - Theta_ref_b)
           + 1/2 (Theta - Theta_ref_b)^T S_b (Theta - Theta_ref_b)
```

and dropping constants independent of ``Theta``:

```text
L_b(Theta) ≈ 1/2 Theta^T S_b Theta - eta_b^T Theta + const
```

with:

```text
eta_b = S_b Theta_ref_b - g_b
```

assuming ``g_b`` is the gradient of the objective with respect to ``Theta`` at
``Theta_ref_b``. If the implementation stores a likelihood score with the
opposite sign convention, the sign must be documented in the schema.

This representation lets the observation-level update combine summaries even
when their linearization points differ:

```text
Lambda_post = Lambda_prior + sum_b S_b
eta_post    = eta_prior    + sum_b eta_b
Theta_post  = solve(Lambda_post, eta_post)
```

The first implementation should still store both forms where practical:

- ``linearization_point`` / ``Theta_ref_b`` for interpretability,
- ``gradient`` / ``g_b`` for local diagnostics,
- ``reduced_information`` / ``S_b`` for curvature,
- ``information_vector`` / ``eta_b`` for accumulation.

## Priors and information-form updates

The observation-level belief state should use information form internally:

```text
Lambda = covariance^{-1}
eta    = Lambda mean
```

A prior belief contributes:

```text
Lambda_prior
eta_prior = Lambda_prior Theta_prior
```

Sub-block summaries contribute:

```text
S_b
eta_b
```

The posterior update is:

```text
Lambda_post = Lambda_prior + sum_b S_b
eta_post    = eta_prior    + sum_b eta_b
Theta_post  = solve(Lambda_post, eta_post)
Cov_post    = inverse_or_pseudoinverse(Lambda_post)
```

This form is convenient because:

- independent block summaries add linearly,
- batches and sequential updates are algebraically equivalent for fixed
  summaries,
- priors naturally regularize weak modes,
- eigenvalue diagnostics are straightforward.

## Synthetic implementation note

The first repository implementation of this layer is a **synthetic
observation-belief demo**. It intentionally stops one level below the expensive
image-backed workflow:

- it constructs a canonical observation-level ``ThetaLayout``,
- it generates synthetic reduced ``SubblockSummary`` objects directly in the
  physical basis,
- it accumulates those summaries against an information-form prior,
- and it emits posterior/eigenmode diagnostics plus machine-readable artifacts.

### Stored summary convention

The synthetic demo uses the following stored names and meanings:

- ``theta_ref``:
  - the physical-basis linearization point ``Theta_ref_b`` for one block
- ``reduced_score``:
  - the objective gradient ``g_b`` with respect to observation-level
    parameters, evaluated at ``theta_ref``
- ``reduced_information``:
  - the Schur-compatible reduced curvature ``S_b`` in the same physical basis

The corresponding information-form contribution is:

```text
eta_b = reduced_information @ theta_ref - reduced_score
```

This sign convention must remain explicit in code and artifacts because future
image-backed paths may naturally produce a score with the opposite sign.

### Why the stored basis is physical

The summary handoff is stored in the canonical physical parameter basis, not in
an eigenbasis, because that basis is:

- stable across blocks,
- interpretable in manifests and CSV products,
- compatible with future selective refresh logic,
- and independent of the conditioning of any particular accumulated posterior.

This keeps the stored summary object durable even when the preferred diagnostic
or solve basis changes.

### Why the observation eigenbasis is diagnostic

The accumulated observation precision may still be eigendecomposed after the
update, but that eigenbasis is treated as a **derived transform**:

- it is built from the current accumulated posterior precision,
- it can change as more summaries are added,
- and it is useful for conditioning and weak-mode reporting.

It should therefore be understood as a diagnostic and optional transform layer,
not as the native persistence format for ``SubblockSummary`` or
``ObservationBeliefState``.

### Why the demo also reports prior-normalized diagnostics

The synthetic demo now reports two complementary diagnostic views:

- physical-basis posterior tables and cumulative histories in native units,
- prior-normalized ratios such as ``posterior_sigma / prior_sigma`` and
  ``|posterior_error| / prior_sigma``.

The physical-basis tables remain the storage and reporting contract because
they preserve the canonical parameter labels and real units. The normalized
ratios are added because the demo mixes arcseconds, log flux, contrast,
arcsec/pixel, and nanometer Zernike coefficients; raw magnitudes alone are not
comparable across those units.

The demo also computes a prior-whitened information-gain matrix

```text
Lambda_gain = diag(prior_sigma) @ S_accum @ diag(prior_sigma)
```

from the accumulated synthetic summary information ``S_accum``. Its eigenmodes
answer a different question from the posterior precision spectrum: which
parameter combinations gained information relative to the prior scale. This is
diagnostic only. The stored summaries and the belief state remain in the
physical basis.

### Difference from the future image-backed path

The synthetic demo proves the accumulator contract without:

- tracing frames,
- rendering images,
- running local optical inference,
- or extracting reduced Fisher products from real sub-block solves.

The later image-backed path should plug into the same interface by replacing the
synthetic ``reduced_information`` / ``reduced_score`` generator with real
Schur-reduced products computed from solved sub-blocks.

## Image-backed sub-block summary export

The first image-backed bridge now exports one real ``SubblockSummary`` from a
prepared observation sub-block context.

### Local parameter split

For one sub-block, the local quadratic is built over a packed parameter vector

```text
[Theta, phi]
```

where:

- ``Theta`` is the enabled observation-level slow/shared state in the canonical
  physical basis,
- ``phi`` is the local fast state used by the sub-block inference recipe,
  initially the packed registration variables.

The exported summary stores only the reduced observation-level block, but the
sidecar artifact also records ``phi_labels``, ``phi_ref``, and the partitioned
curvature blocks.

### Reference glossary

The smoke workflow now keeps these references explicit in
``schur_summary_plan.json`` and ``schur_summary_audit.json``:

- truth trace: simulated frame-level values used to render the cube,
- optimizer initialization: active-state values where registration inference
  starts if it runs,
- ``phi_ref``: fast-state point used to linearize the Schur summary,
- recovered reference: a ``phi_ref`` obtained from registration inference,
- ``preconditioning_reference``: point used to build optimizer
  preconditioning,
- ``theta_ref``: slow observation-level point used to linearize the summary,
- observation prior mean: belief mean used by the observation update, defaulting
  to summary ``theta_ref`` in real-summary mode.

### Score and curvature convention

Around a reference point ``(Theta_ref, phi_ref)``, the exporter computes the
local objective gradient and curvature

```text
g = [g_theta, g_phi]

H = [[H_tt, H_tp],
     [H_pt, H_pp]]
```

and then eliminates the fast block with

```text
S = H_tt - H_tp solve(H_pp, H_pt)
g_reduced = g_theta - H_tp solve(H_pp, g_phi)
```

The exported ``SubblockSummary`` therefore represents

```text
L_b(theta) ~= const
            + g_reduced.T @ (theta - theta_ref)
            + 0.5 * (theta - theta_ref).T @ S @ (theta - theta_ref)
```

using the same sign convention as the synthetic accumulator demo: the stored
``reduced_score`` is the objective gradient at ``theta_ref``.

### Dense and structured curvature paths

The dense Schur-summary exporter remains the small-case validation bridge: it
builds one packed ``[Theta, phi]`` objective, differentiates a dense local
gradient/Hessian, partitions the blocks, and writes the existing
``SubblockSummary`` artifact contract.

The first structured Schur-summary path now supports the independent-frame,
registration-only case with frame-local active state and no active shared
sub-block state. It builds per-frame local quadratics over ``[Theta, phi_i]``
and applies the same ``subblock_reduce`` policy as the image objective:
``sum`` uses one unit-weighted contribution per frame, while ``mean`` uses
``1 / n_frame`` per frame. The Schur reduction is then performed frame by
frame:

```text
S = sum_i weight * (H_tt_i - H_tphi_i solve(H_phiphi_i, H_phitheta_i))
g_reduced = sum_i weight * (g_theta_i - H_tphi_i solve(H_phiphi_i, g_phi_i))
```

Exporter and consumer policy separates optimizer scaling from observation
information units. An optimizer objective may use ``subblock_reduce: mean``,
while default Schur export records ``summary_information_scale:
summed_likelihood`` and uses summed reduction for handoff. Real-summary
forecast and belief-update consumers require that metadata by default;
optimizer-scale summaries remain available only through an explicit
legacy/debug consumer opt-in.

``run_obs_subblock_study.py --mode schur_summary`` exposes
``--schur-curvature-method auto|dense|structured_independent_frames``. ``auto``
uses the dense path below ``--max-dense-dim`` and switches to the structured
independent-frame path when the packed dense guard would otherwise fail and the
layout is supported. The default dense guard is now 40, so the 20-frame
four-scalar registration-only layout with ``combined_dim = 64`` uses the
structured path in auto mode. Dense-vs-structured comparison is validation-only
and must be requested explicitly with ``--validate-structured-against-dense``.

For compatibility, the first structured exporter still writes dense
``H_tt/H_tp/H_pp`` sidecar arrays materialized from the structured blocks for
small and medium validation cases. The expensive operation avoided is dense
autodiff over the full packed vector. This sidecar policy is transitional and
does not change ``load_subblock_summary``.

Structured Schur summaries also carry frame-quality metadata in diagnostics.
Recovered-reference per-frame reduced chi-squared values can drive
``warn``/``mask``/``reject`` export policies. Under ``mask`` the reduced
information and score stored in ``SubblockSummary`` already reflect only the
included frames, with compact diagnostics such as the policy, good/bad frame
counts, bad-frame indices, threshold, and effective frame fraction. The
observation-level accumulator can consume these summaries normally; frame
quality affects the summary's reduced ``S_b``/``g_b`` before handoff rather than
changing the core summary API.

### First validation workflow

The recommended first manual validation is still one tiny sub-block, not a
campaign:

1. run ``run_obs_subblock_study.py --mode schur_summary`` on a 3-frame
   noiseless registration-only case using
   ``--theta-keys source.separation_as,source.log_flux_total,source.contrast,optics.plate_scale_as_per_pix`` and
   ``--phi-ref truth_when_available``,
2. inspect ``subblock_summary.json`` with
   ``examples/scripts/inspect_subblock_summary.py``,
3. check the Schur and surrogate-validation artifacts,
4. run ``run_observation_belief_update_demo.py --summary-path ...`` on that
   one exported summary.

This answers the first practical bridge questions:

- does the image-backed exporter write a numerically sane reduced summary,
- does the reduced quadratic have the expected local sign behavior,
- and can the observation-level accumulator consume the real artifact without
  special-case bookkeeping.

The earlier dense-autodiff block on ``source.log_flux_total`` and
``source.contrast`` was an implementation limitation, not a claim about those
parameters. The failing path differentiated through full
``ParameterStore.refresh_derived(...)``, which reached source-photometry
transforms that used Python ``float(...)`` coercion. The current Schur-summary
objective now follows the canonical active-parameter semantics more closely:
start from a resolved base store outside autodiff, overlay active Theta values
directly, and repair only the minimal dependent source photometry quantity
(``source.raw_fluxes``) with JAX-safe array operations.

The two-key case
``source.separation_as,optics.plate_scale_as_per_pix`` remains a useful
fallback when debugging the smallest possible dense-Hessian path.

### Real-summary prior initialization

When the observation-level updater consumes real image-backed
``subblock_summary.json`` artifacts, its default prior mean should come from the
same effective context that produced those summaries. The current default policy
is:

1. explicit prior config or preset wins when the user supplies one;
2. otherwise use the summary ``theta_ref`` values as the default prior mean;
3. otherwise reconstruct from summary-resolved system context when available;
4. only then fall back to the bare default preset, with a warning.

This is a provenance rule, not an exposure-time-specific patch. A bare
``SHERA_FLIGHT_3P`` preset can be stale for real summaries because the
trace/render/inference path may have applied non-default context such as short
frame exposure time, source overrides, or other runtime assumptions. The
summary ``theta_ref`` is the safest default because it is the actual
linearization point used in the exported quadratic.

The policy is now implemented as shared inference helper code in
``dluxshera.inference.observation_forecast`` rather than being owned by the
demo script. Both ``run_observation_belief_update_demo.py`` and
``run_observation_summary_simulator.py`` should use that shared helper so
future forecast modes inherit the same prior-context provenance behavior.

### Observation summary simulator

``examples/scripts/run_observation_summary_simulator.py`` is the first
forecast-oriented harness above the image-backed summary handoff. It consumes
one or more existing ``SubblockSummary`` artifacts, validates that they share
identical ``theta_labels``, resolves the prior mean with the same real-summary
policy as the belief-update demo where practical, and then runs the existing
observation-level update for a requested grid of accumulated sub-block counts.

The initial ``replicate`` mode is deliberately deterministic: one source
summary is repeated, or multiple source summaries are tiled in input order and
truncated. This is an accumulation sanity check, not a realism model. It does
not add score noise, bootstrap Monte Carlo outputs, sample matrix entries, or
condition summaries on trajectory context.

The first stochastic mode is ``fixed_information_score_noise``. It preserves
each template summary's ``theta_ref`` and reduced information matrix while
drawing independent reduced-score vectors around
``g_expected = S @ (theta_ref - theta_true)`` with covariance ``alpha * S``.
The mode uses nested prefixes across the requested sub-block grid, records the
truth vector and score-noise settings in the manifest, and reports trial-level
posterior means/errors plus aggregate separation RMS error. The score-noise
scale ``alpha`` is not yet calibrated by real summary Monte Carlo.

``examples/scripts/run_obs_subblock_monte_carlo.py`` now provides that empirical
calibration path. It repeatedly generates image-backed Schur-summary artifacts
from real observation sub-block renders, aggregates ``S_b`` and ``g_b`` metrics,
and writes whitened score-residual diagnostics plus
``aggregate/accepted_summary_paths.csv``. Those outputs should be used to
calibrate score-noise scale choices and to seed future bootstrap-style summary
synthesis modes; the Monte Carlo runner itself remains below the simulator and
does not perform observation-level forecast updates.

The first forecast metric is ``source.separation_as`` posterior sigma in
microarcseconds versus accumulated sub-block count. The simulator writes a
manifest, CSV tables, matrix diagnostics, and a
``separation_sigma_vs_n_subblocks.png`` plot under
``Results/observation_summary_simulator/<run-name>/``.

Future synthesis modes may include bootstrap sampling from real summary outputs
and trajectory-conditioned summaries. Those extensions should not change the
role of this harness: it forecasts behavior from already-exported summaries and
does not replace image-backed Schur summary validation. The shared prior-context
helper boundary is intended to keep those future modes from importing
demo-script internals.

## Parameter layout and group toggles

The observation-level layer should define an explicit ``ThetaLayout``. This is a
physical-basis layout, not an eigenmode layout.

A first YAML sketch:

```yaml
observation_update:
  theta_layout:
    source:
      separation_as: true
      contrast: true
      log_flux_total: true
      position_angle_deg: false
    optics:
      plate_scale_as_per_pix: true
      primary_zernikes:
        enabled: true
        indices: [0, 1, 2, 3, 4, 5]
      secondary_zernikes:
        enabled: true
        indices: [0, 1, 2, 3, 4, 5]
```

Early runs can disable groups without changing the summary contract:

```yaml
observation_update:
  theta_layout:
    optics:
      primary_zernikes:
        enabled: false
      secondary_zernikes:
        enabled: false
```

This lets the first implementation keep the full canonical parameter family in
mind while still allowing small test cases.

## Interaction with sub-block active state

The sub-block inference active state and the observation-level ``ThetaLayout``
are related but not identical.

A sub-block inference configuration may define:

```yaml
active:
  frame_keys:
    - source.x_position_as
    - source.y_position_as
    - source.position_angle_deg
  shared_keys: []
```

for the first registration-only demo.

A later configuration may define:

```yaml
active:
  frame_keys:
    - source.x_position_as
    - source.y_position_as
    - source.position_angle_deg
  shared_keys:
    - optics.plate_scale_as_per_pix
```

In the second case, ``optics.plate_scale_as_per_pix`` is active in the local
sub-block solve. The observation-level layer then has a design choice:

- include plate scale in ``Theta`` and use the block solve as part of a joint
  slow-state update,
- or exclude plate scale from ``Theta`` and treat it as a block-local nuisance
  term ``psi_b`` that is eliminated in the Schur complement,
- or collect its recovered blockwise values as an auxiliary observation product.

The implementation should therefore derive the Schur partition from two
explicit lists:

```text
observation_theta_keys
local_eliminated_keys
```

rather than assuming that all sub-block shared keys must be propagated.

## Observation-level basis and eigenmode strategy

Eigenmodes should be treated as a **transform layer**, not as the native storage
basis for the belief state.

The native belief state should remain in the physical canonical parameter basis:

```text
source.separation_as
source.contrast
source.log_flux_total
optics.plate_scale_as_per_pix
optics.primary.zernike_coeffs_nm[...]
optics.secondary.zernike_coeffs_nm[...]
```

The optional observation-level eigenbasis is derived from the accumulated
precision / Fisher matrix:

```text
Lambda = V diag(lambda) V^T
```

and defines a coordinate transform such as:

```text
DeltaTheta = V z
```

or, for a whitened parameterization:

```text
DeltaTheta = V diag(1 / sqrt(lambda)) z
```

This mirrors the canonical eigenmode idea but should be implemented as an
observation-level transform. It may borrow principles from the existing
``EigenThetaMap`` used by canonical recipes, but the exact class may not be
reusable without adaptation because the observation-level update operates on
summary information rather than directly on a single image-domain objective.

### Why observation-level eigenmodes matter

Observation-level eigenmodes are expected to be important for:

- diagnosing degeneracies between M1 and M2 Zernike coefficients,
- identifying which linear combinations of optical terms are actually measured,
- regularizing or truncating weak modes,
- reporting constrained and unconstrained directions separately,
- and stabilizing the solve when the full canonical parameter set is enabled.

For example, the accumulated Fisher may show that one combination of M1 and M2
Zernike terms is strongly constrained while an orthogonal combination remains
weak. The eigenbasis should expose this directly.

### Distinguish sub-block and observation eigenbases

There are two different eigenmode bases that should remain conceptually
separate.

#### Sub-block eigenbasis

A sub-block eigenbasis would reparameterize the local optimization variables:

```text
[frame-local keys, block-shared keys]
```

This is useful for convergence of local sub-block inference, especially when
shared terms are added to the active state. It is not required for the first
registration-only demo if that solve already converges well.

#### Observation-level eigenbasis

An observation-level eigenbasis reparameterizes the accumulated slow-state
update:

```text
Theta
```

This is useful for conditioning, reporting, and degeneracy analysis of the
observation-level belief state. It can be implemented even if the local
sub-block solves remain in primitive registration coordinates.

### Suggested observation eigenbasis object

A future object may look like:

```text
ObservationThetaMap
  labels: tuple[str, ...]
  theta_ref: array
  basis_vectors: array      # columns or rows must be documented
  eigenvalues: array
  whitening: bool
  retained_mask: array

  physical_to_eigen(theta)
  eigen_to_physical(z)
  physical_delta_to_eigen(delta)
  eigen_delta_to_physical(z)
```

This should be designed around physical labels and accumulated precision
matrices, not around a particular image-domain model function.

## First-draft schema: `SubblockSummary`

### Purpose

``SubblockSummary`` is the local handoff artifact emitted by one solved
sub-block. It should carry enough information that the observation-level layer
can reason about the slow/shared state without immediately revisiting raw frame
optimization for every block.

### Design goals

A ``SubblockSummary`` should be:

- locally meaningful on its own,
- cheap to store relative to raw full optimization state,
- explicit about the state partition used during the local solve,
- explicit about the shared-state point around which its summary is valid,
- rich enough to support:
  - observation-level slow/shared updates,
  - summary-vs-refresh decisions,
  - post-hoc diagnostics and debugging.

### Conceptual fields

#### 1. Identity and provenance

These fields identify the block and the assumptions under which it was solved.

- ``block_id``
- ``observation_id``
- ``time_start_s``
- ``time_end_s``
- ``n_frame``
- ``input_cube_path`` or manifest reference
- ``shared_state_hash`` or equivalent fingerprint
- optional config snapshot / manifest pointer

#### 2. State partition

These fields define how the local and observation-level variables were treated.

- ``frame_keys``
- ``block_shared_active_keys``
- ``observation_theta_keys``
- ``local_eliminated_keys``
- ``reported_auxiliary_keys``

This is important because the same physical parameter may be active locally in
one study but propagated at the observation level in another.

#### 3. Local solution

These fields describe what the local solve actually found.

- recovered fast state
- recovered shared block state, if any
- objective value at optimum
- objective decomposition:
  - data term
  - prior term
  - temporal term
- optimizer metadata:
  - iteration count
  - status flags
  - convergence summary

#### 4. Local diagnostics

These fields help determine whether the block fit is healthy and whether the
summary remains trustworthy later.

Examples include:

- reduced-chi-squared-like scalar
- standardized residual RMS
- framewise data-term summaries
- fit warning flags
- representative image-fit diagnostics path references

#### 5. Schur-reduced sensitivity products

These fields are the most directly relevant to the observation-level estimator.

Recommended fields:

- ``linearization_point`` / ``Theta_ref_b``
- ``objective_gradient`` / ``g_b``
- ``reduced_information`` / ``S_b``
- ``information_vector`` / ``eta_b``
- Schur complement diagnostics:
  - local nuisance rank
  - condition number
  - damping / eigenvalue floor used
  - discarded or weak modes
- cross-coupling and absorption diagnostics

#### 6. Validity / refresh metadata

These fields help decide whether the block should later be revisited.

Examples include:

- shared-state point around which the summary was linearized
- local trust-region scale
- predicted sensitivity to selected slow/shared parameters
- refresh priority hint
- summary version / schema version

### Minimal first version

A realistic minimal first version could include:

- identity / provenance
- state partition
- local solution objective decomposition
- standardized residual summary
- common or per-block linearization point
- Schur-reduced information matrix for enabled observation-level parameters
- reduced gradient or information vector
- basic Schur diagnostics

That would already be enough to support early observation-level studies.

### Example YAML sketch

```yaml
SubblockSummary:
  schema_version: "subblock_summary.v0"
  block_id: "block_0007"
  observation_id: "obs_demo_001"

  timing:
    time_start_s: 6.0
    time_end_s: 7.0
    n_frame: 20

  inputs:
    cube_path: "render/block_0007_cube.fits"
    manifest_path: "render/manifest.json"

  state_partition:
    frame_keys:
      - source.x_position_as
      - source.y_position_as
      - source.position_angle_deg
    block_shared_active_keys: []
    observation_theta_keys:
      - source.separation_as
      - source.contrast
      - source.log_flux_total
      - optics.plate_scale_as_per_pix
      - optics.primary.zernike_coeffs_nm[0]
      - optics.secondary.zernike_coeffs_nm[0]
    local_eliminated_keys:
      - frame[*].source.x_position_as
      - frame[*].source.y_position_as
      - frame[*].source.position_angle_deg
    reported_auxiliary_keys: []

  local_solution:
    objective_total: 1243.81
    objective_terms:
      data: 1243.81
      prior: 0.0
      temporal: 0.0
    optimizer:
      status: "converged"
      n_iter: 76

  diagnostics:
    zscore_rms: 1.08
    reduced_chi2: 1.12
    flags: []

  schur_summary:
    basis: "physical"
    sign_convention: "objective_gradient"
    labels:
      - source.separation_as
      - source.contrast
      - source.log_flux_total
      - optics.plate_scale_as_per_pix
      - optics.primary.zernike_coeffs_nm[0]
      - optics.secondary.zernike_coeffs_nm[0]
    linearization_point:
      values: [8.995, 3.02, 12.4, 0.0301, 0.0, 0.0]
    objective_gradient:
      values: [0.4, -0.1, 0.02, 2.5, -0.03, 0.08]
    information_vector:
      values: [1124.1, 21.2, -0.5, 31.8, 0.7, -1.3]
    reduced_information:
      storage: "dense"
      matrix:
        - [125.3, -2.1, 0.0, 4.2, 0.3, -0.2]
        - [-2.1, 84.9, 0.1, -1.7, 0.0, 0.1]
        - [0.0, 0.1, 12.0, 0.4, 0.2, 0.2]
        - [4.2, -1.7, 0.4, 77.0, -0.8, 1.1]
        - [0.3, 0.0, 0.2, -0.8, 3.4, -3.2]
        - [-0.2, 0.1, 0.2, 1.1, -3.2, 3.3]
    diagnostics:
      nuisance_rank: 60
      nuisance_condition_number: 2.1e7
      schur_damping: 1.0e-8
      min_reduced_eigenvalue: 1.0e-4
      max_reduced_eigenvalue: 225.0

  validity:
    refresh_priority_hint: 0.08
    trust_radius_metric: "mahalanobis"
    notes: "summary valid near current shared-state point"
```

## First-draft schema: `ObservationBeliefState`

### Purpose

``ObservationBeliefState`` is the rolling slow/shared state estimate used across
many sub-blocks in one observation.

It is not simply a copy of the latest ``SubblockSummary``. Instead, it
represents the **accumulated observation-level estimate** built from many block
contributions.

### Design goals

``ObservationBeliefState`` should:

- remain low-dimensional relative to the total local block state,
- explicitly represent uncertainty,
- store and report state in the physical canonical parameter basis,
- optionally include an eigenmode transform for solving and diagnostics,
- be compatible with future filtering or smoothing interpretations,
- provide enough metadata to determine:
  - what blocks contributed,
  - what shared-state point is currently assumed,
  - what refresh policies have been triggered.

### Conceptual fields

#### 1. Identity / span

- ``observation_id``
- processed block range
- time span covered
- schema version

#### 2. Theta layout

- physical labels
- enabled parameter groups
- units
- reference values
- active/excluded group notes

#### 3. Shared-state estimate

- mean vector for selected observation-level parameters
- covariance and/or precision
- information vector
- optional cross-covariances if useful

#### 4. Eigenbasis diagnostics

- eigenvalues of the accumulated precision
- eigenvectors in the physical parameter basis
- retained / truncated mode mask
- whitening convention
- weak-mode diagnostics

#### 5. Update provenance

- list or log of contributing blocks
- timestamps or block indices of major updates
- latest update method / approximation class
- prior source or initialization source

#### 6. Consistency / health tracking

- aggregate observation objective surrogate
- flagged blocks
- current trust-region or validity metadata
- latest refresh decision or refresh mode

### Minimal first version

A realistic first ``ObservationBeliefState`` can remain simple:

- observation identifier
- physical theta labels
- prior mean / precision
- accumulated information vector
- accumulated precision
- posterior mean / covariance
- contributing block count
- eigenvalue diagnostics
- flagged block list
- update history notes

This is enough for early testing of observation-level update logic without
prematurely committing to a final filtering formalism.

### Example YAML sketch

```yaml
ObservationBeliefState:
  schema_version: "observation_belief_state.v0"
  observation_id: "obs_demo_001"

  span:
    first_block_id: "block_0000"
    last_block_id: "block_0099"
    n_blocks_processed: 100

  theta_layout:
    basis: "physical"
    labels:
      - source.separation_as
      - source.contrast
      - source.log_flux_total
      - optics.plate_scale_as_per_pix
      - optics.primary.zernike_coeffs_nm[0]
      - optics.secondary.zernike_coeffs_nm[0]
    enabled_groups:
      source: true
      plate_scale: true
      primary_zernikes: true
      secondary_zernikes: true
    units:
      source.separation_as: "arcsec"
      source.contrast: "dimensionless"
      source.log_flux_total: "log flux"
      optics.plate_scale_as_per_pix: "arcsec / pixel"
      optics.primary.zernike_coeffs_nm[0]: "nm"
      optics.secondary.zernike_coeffs_nm[0]: "nm"

  information_state:
    prior_mean:
      values: [8.995, 3.02, 12.4, 0.0301, 0.0, 0.0]
    prior_precision:
      storage: "dense"
      matrix: "..."
    accumulated_information_vector:
      values: "..."
    accumulated_precision:
      storage: "dense"
      matrix: "..."
    posterior_mean:
      values: [9.001, 3.01, 12.38, 0.03005, 0.2, -0.1]
    posterior_covariance:
      storage: "dense"
      matrix: "..."

  eigenbasis:
    enabled: true
    source_matrix: "posterior_precision"
    whitened: true
    eigenvalues: [225.0, 120.0, 40.0, 3.0, 0.08, 0.001]
    retained_mask: [true, true, true, true, false, false]
    weak_mode_notes:
      - "weak combination dominated by primary/secondary Zernike common mode"

  provenance:
    initialized_from: "system config / prior belief"
    update_mode: "batch"
    contributing_blocks:
      - "block_0000"
      - "block_0001"
      - "..."
      - "block_0099"
    notes:
      - "first observation-level update applied after collecting 100 summaries"

  consistency:
    flagged_blocks: []
    refresh_mode: "none"
    aggregate_surrogate_nll: 182.7
```

## Relationship between the two schemas

The conceptual relationship is:

- ``SubblockSummary`` is **local** and block-specific,
- ``ObservationBeliefState`` is **global** within one observation and
  accumulates across blocks.

In other words:

- many ``SubblockSummary`` objects inform one evolving
  ``ObservationBeliefState``,
- the current ``ObservationBeliefState`` supplies the assumed shared state used
  by future local sub-block solves.

That feedback loop is the core hierarchical cascade.

## Refresh / revisit policy

A central design question is when previously solved sub-blocks should be
revisited after the observation-level belief changes.

The guiding principle is:

> revisit a block when its cached local approximation is no longer trustworthy,
> not merely because the belief state changed.

Three trigger families are useful.

### 1. Parameter-space trigger

A slow/shared update may move too far from the point about which a block summary
was computed.

This is naturally a trust-region style trigger:

- small update -> reuse existing summary,
- moderate update -> selectively refresh affected blocks,
- large update -> broader refresh may be required.

A simple first diagnostic is the Mahalanobis displacement between the updated
belief and a block's linearization point:

```text
d_b^2 = (Theta_post - Theta_ref_b)^T W_b (Theta_post - Theta_ref_b)
```

where ``W_b`` may initially be the block's reduced information matrix, a prior
metric, or a diagonal scale matrix.

### 2. Data-space trigger

Even when parameter-space movement appears small, the data-space diagnostics may
show that the current local surrogate is inadequate.

Examples include:

- persistent standardized residual structure,
- elevated reduced-chi-squared-like values,
- mismatch between predicted and actual block objective behavior.

### 3. Consistency trigger

The accumulated observation-level state and the cached local summaries may
become mutually inconsistent.

Examples include:

- multiple neighboring blocks show coherent tension with the current belief,
- predicted local corrections fail repeatedly,
- summary validity metadata indicates that a local approximation is being used
  outside its intended region.

## Refresh modes

The architecture should support three refresh modes.

### No refresh

Continue using cached sub-block summaries.

### Selective refresh

Warm-start and re-solve only the most affected blocks.

This is likely to be the most important practical mode.

### Global refresh

Re-solve all sub-blocks under the updated belief state.

This should be treated as available, but expensive.

## Refresh decision philosophy

The refresh policy should eventually answer:

> Is the current set of cached local summaries still an adequate surrogate for
> the observation-level objective under the updated shared-state belief?

That is the right high-level decision criterion.

A future implementation may realize that decision using a combination of:

- parameter-space displacement,
- residual-based diagnostics,
- predicted correction magnitude from local cross-coupling,
- consistency checks across neighboring blocks.

This note does not prescribe exact thresholds yet.

## Why a full observation simulation is not required immediately

A full mission-like observation could contain around 1,800 sub-blocks for a
30-minute observation under the current working assumptions.

That is **not** the right first demonstration target.

The architecture should first be validated on the smallest synthetic hierarchy
that still exercises the cascade:

- local solve,
- Schur summary handoff,
- observation-level update,
- eigenvalue / conditioning diagnostics,
- refresh decision.

This means the first honest demonstrations can be much smaller, for example:

- 1 block,
- 3 blocks,
- 5 blocks,
- 10 blocks,

rather than 1,800.

A moderate scaling demonstration can then use 20--100 one-second sub-blocks
without requiring a full 30-minute observation.

## Proposed first demonstration

The first demonstration should target:

```text
Observation-level belief update from Schur-complement Fisher summaries
emitted by registration-only sub-block solves.
```

Recommended initial setup:

- ``n_subblocks``: 10 initially, then 100
- ``frames_per_subblock``: 20
- ``subblock_duration_s``: 1
- ``frame_rate_hz``: 20
- frame truth:
  - iid X/Y jitter at roughly 120 mas
  - iid PA jitter at roughly 2e-3 deg
- sub-block active state:
  - ``source.x_position_as``
  - ``source.y_position_as``
  - ``source.position_angle_deg``
- observation theta layout:
  - full canonical parameter family available
  - Zernike groups configurable on/off

The first tests may disable Zernikes, but the schema and implementation should
not be scalar-only. The design should support the full canonical parameter set
from the beginning.

## Validation ladder

A staged validation ladder is recommended.

### Stage 1: single-block Schur summary validity

Goal:
- determine whether the Schur summary captures locally useful slow/shared
  sensitivity.

Procedure:
- solve one block,
- save its summary,
- perturb one or more candidate observation-level parameters slightly,
- compare:
  - predicted change from the summary,
  - actual change from a refreshed solve.

Success criterion:
- the summary predicts local behavior well enough to justify carrying it
  upward.

### Stage 2: tiny toy observation

Goal:
- demonstrate that multiple block summaries can update a slow/shared belief
  coherently.

Procedure:
- generate 3--10 synthetic blocks under a controlled slow/shared mismatch,
- solve them under an initial imperfect belief,
- combine their Schur summaries into an observation-level update,
- compare updated belief to truth.

Success criterion:
- observation-level update moves the belief in the right direction and with
  sensible uncertainty behavior.

### Stage 3: full-layout disabled-group test

Goal:
- verify that the full canonical layout can be constructed while some groups are
  disabled.

Procedure:
- define the full ``ThetaLayout``,
- disable primary/secondary Zernike groups,
- run the observation update on the remaining enabled groups,
- verify labels, slicing, prior blocks, and summary accumulation are stable.

Success criterion:
- turning parameter groups on/off does not require code-path changes.

### Stage 4: selected Zernike test

Goal:
- test the first M1/M2 degeneracy handling at the observation level.

Procedure:
- enable one or a small number of primary and secondary Zernike components,
- accumulate Schur summaries,
- eigendecompose the accumulated precision,
- inspect weak and strong modes.

Success criterion:
- eigenmode diagnostics expose constrained and weakly constrained optical
  combinations.

### Stage 5: selective refresh demonstration

Goal:
- demonstrate that selective refresh is a plausible computational compromise.

Procedure:
- update the slow/shared belief after combining summaries,
- rank blocks by predicted sensitivity / tension,
- selectively refresh only the top few blocks,
- compare against a full refresh of all blocks.

Success criterion:
- selective refresh recovers most of the benefit of full refresh at lower cost.

### Stage 6: moderate scaling study

Goal:
- characterize how the approach behaves as the number of blocks grows.

Procedure:
- repeat the toy-observation idea on larger but still manageable stacks
  (for example 20--100 blocks).

Success criterion:
- summary-based updates remain stable and refresh frequency remains plausible.

## Candidate first parameters for observation-level studies

This note does not decide the final parameter allocation, but useful candidates
for early study include:

- ``source.separation_as``
- ``source.contrast``
- ``source.log_flux_total``
- ``optics.plate_scale_as_per_pix`` when it is not estimated locally at the
  sub-block level
- a configurable set of low-order aberration terms

The main question is not “which parameter is most interesting,” but:

> which parameters are sufficiently detectable and estimable at the
> observation-level scale to justify inclusion in the rolling belief state and
> eventual observation-level updates?

## Immediate implementation implications

This design suggests several staged follow-ons:

1. define ``ThetaLayout`` / physical-basis observation parameter packing,
2. enrich sub-block output artifacts toward ``SubblockSummary``,
3. compute Schur-complement Fisher summaries for enabled observation-level
   parameters,
4. support explicit state partitions:
   - frame-local keys,
   - block-shared active keys,
   - observation theta keys,
   - locally eliminated keys,
5. store per-block linearization points and information vectors,
6. define an ``ObservationBeliefState`` container for toy observation-level
   studies,
7. implement a minimal summary-combining observation-level prototype,
8. add optional observation-level eigenbasis diagnostics,
9. add selective refresh experiments on tiny synthetic observations.

## Non-goals for the first observation-level prototype

The first prototype should explicitly avoid:

- a full mission-scale recursive filter,
- solving 1,800 sub-blocks as a prerequisite,
- final storage formats for large matrix products,
- final flight-style runtime policies,
- requiring local sub-block eigenmode optimization before observation-level
  eigenmode diagnostics are explored.

## Open questions

The following remain intentionally open:

- Which slow/shared parameters should first enter the observation-level belief
  state?
- Which parameters should instead be estimated as sub-block shared active terms?
- How should locally estimated block-shared parameters be aggregated when they
  are not part of ``Theta``?
- Which sub-block sensitivity products are the smallest useful handoff set?
- What is the best practical surrogate for standardized residual health?
- What refresh ranking signals work best in practice?
- How often will selective refresh be sufficient relative to global refresh?
- What minimal schema versioning and artifact references will be most useful for
  reviewability and reproducibility?
- Should the existing canonical ``EigenThetaMap`` be generalized, wrapped, or
  mirrored by a dedicated observation-level map?

## Summary

The observation-level estimation problem should be treated as a reduced
hierarchical inference layer built on top of local sub-block solves.

The key architectural move is to define explicit handoff objects:

- ``SubblockSummary`` for local block evidence, Schur-reduced information, and
  validity metadata,
- ``ObservationBeliefState`` for the accumulated slow/shared estimate.

The preferred sensitivity handoff is a Schur-complement Fisher summary. This
allows the sub-block layer to eliminate frame-local registration variables and
any block-shared nuisance variables before passing information upward.

The full canonical parameter family should be represented from the beginning by
a configurable physical-basis ``ThetaLayout``. Early tests can disable groups,
but the implementation should not be scalar-only.

Observation-level eigenmodes should be treated as an optional transform layer
built from the accumulated precision matrix. They are expected to be important
for diagnosing and regularizing M1/M2 Zernike degeneracies, but they should not
replace the physical-basis storage and reporting contract.

The immediate next goal is not mission-scale execution. It is to implement and
validate the smallest honest version of this hierarchy on tiny synthetic
observations, then scale to 20--100 sub-blocks once the summary and update
contracts are behaving sensibly.

## Observation Bias Campaign

`examples/scripts/run_observation_bias_campaign.py` is the first small
observation-level campaign layer for image-backed summaries. It is deliberately
not a mission-scale runner: the default smoke shape is a few sub-blocks with a
few frames each, and it reuses `run_obs_subblock_study.py --mode schur_summary`
for trace/render/objective/Schur work.

The campaign derives its observation `Theta` layout from the resolved system
store by default. Source scalars, plate scale, and all primary/secondary
Zernike coefficients present in the store are physical-basis labels. Mask
controls can include or exclude Zernike subsets for smoke tests without
changing the storage convention.

The native belief state remains in physical labels. Observation eigenmodes are
diagnostic transforms of either accumulated summary information or posterior
precision. They are used to expose constrained and weak optical combinations,
especially paired M1/M2 Zernike directions, and should not be treated as the
canonical storage basis.

Campaign success should be read through two lenses:

- whether weak and strong optical combinations are visible in the eigenmode
  tables,
- whether `source.separation_as` recovery behaves sensibly under biased
  reference/prior states.

It is not a requirement that every individual M1/M2 coefficient be recovered
independently in the physical basis.
