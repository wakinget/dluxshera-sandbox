# Observation Sub-Block Inference Design Note

## Purpose

This note defines the intended direction for observation sub-block inference.

The immediate implementation goal remains a readable first block-inference recipe
that jointly solves for frame-varying registration parameters over a short stack
of frames. However, this document is intentionally broader than that first
milestone. It is meant to describe the longer-term mission-aligned inference
story so that the initial recipe, schema, and artifacts grow in the right
direction.

The central design goal is to treat sub-block inference as a structured
time-domain estimation problem rather than as a one-off optimization over a
small synthetic dataset.

## Context

We now have a working sub-block workflow with three main pieces:

- a trace-generation stage that produces canonical explicit frame traces
- a rendering stage that produces an observation sub-block FITS cube from a
  shared base state plus per-frame updates
- an inference stage that begins by fitting per-frame registration terms from
  the rendered cube

The current prototype is intentionally narrow: it holds the shared system state
fixed and fits only per-frame registration. That is a useful first milestone,
but it is not the full intended mission inference story.

A more realistic mission-level view is:

- the instrument carries a time-evolving belief about many parameters
- some parameters are effectively static over long spans
- some parameters vary slowly from block to block
- some parameters vary frame to frame
- image data update that belief over time

This note therefore distinguishes between the first implemented recipe and the
broader estimation architecture we intend to support.

## Operational framing

The current understanding of the mission concept is that full frames are not the
primary downlinked product. Instead, an on-board centroiding process identifies
a region of interest (ROI) around the PSF, and that ROI remains fixed for a
short interval before being re-centered.

A useful working assumption is:

- frame cadence is approximately 20 Hz
- the ROI remains fixed for roughly 1 second
- a new ROI is chosen approximately every 1 second
- each fixed-ROI interval therefore contains about 20 frames

This naturally defines an observation sub-block as the interval over which one
ROI crop remains fixed.

That structure strongly motivates a hybrid estimation strategy:

- **within one sub-block:** use a joint block fit / smoothing-style inference
  over all frames in the block
- **across successive sub-blocks:** propagate a running belief state forward in
  time, updating it as each new block arrives

This note adopts that hybrid view as the long-term design direction.

## Design goal

The long-term goal is to perform inference on a sequence of observation
sub-blocks using:

- a forward image model
- a partition between shared and frame-varying latent state
- priors or state uncertainty for all important modeled quantities
- temporal structure for the frame-varying state
- a mechanism for carrying information from one sub-block to the next

The first recipe does not need to implement all of that, but it should be
understood as the first restricted case of that broader problem.

## Terminology

To avoid conflating several different ideas, this note uses the following terms.

### Belief state

The mission’s current estimate of parameter values and uncertainty.

This may include nearly every important modeled quantity, even when a given
parameter is not actively varied in the current solve. Examples include:

- binary separation
- contrast or total flux terms
- plate scale
- selected optical aberration terms
- detector calibration state
- registration anchors or slowly drifting pointing state

A parameter may live in the belief state without being part of the active
optimization variables for a particular sub-block.

### Active inference state

The subset of parameters that are actually varied during the current solve.

Examples include:

- frame-varying `x/y/PA` in the first phase
- later, one or more shared parameters such as plate scale
- later still, selected slowly varying calibration or optical terms

Only parameters in the active inference state participate directly in the
current optimization objective.

### Frozen state

Parameters that are held fixed during the current solve.

These parameters may still be uncertain in the broader belief state, but that
uncertainty does not automatically affect the local solve unless we explicitly
propagate or model it.

Examples may include:

- detector calibration maps
- contrast in a registration-only run
- higher-order optical terms in an early simplified solve

### Prior

A probabilistic constraint on an actively inferred parameter or process.

In this note, “prior” is not treated as a synonym for “belief state.” A prior
is the part of the belief that is actively used in the current inference
problem.

This distinction matters because a frozen-but-uncertain parameter may exist in
the belief state without appearing as an active prior term in the current
objective.

### Temporal model

A model that links the state between frames or between sub-blocks.

Examples include:

- independent per-frame parameters
- anchor plus residual parameterization
- linear drift plus jitter
- random walk
- more structured stochastic processes in future work

The temporal model is expected to become increasingly important as the inference
problem becomes more realistic.

## Why this distinction matters

A common conceptual trap is to treat every modeled quantity as if it either:

- has a prior and is therefore being inferred, or
- is fixed and therefore irrelevant to uncertainty propagation

Neither of those is quite right.

A better way to think about the problem is:

- nearly every important parameter may belong to the mission belief state
- only some parameters are active in a given solve
- priors apply to the active subset
- frozen uncertain parameters may still need to be handled through joint
  inference, marginalization, or approximate uncertainty propagation

This gives us a more realistic long-term path than treating the first
registration-only recipe as the final architecture.

## Inference architecture

The intended architecture is hierarchical.

### Within a sub-block

For frames that share one fixed ROI crop, inference should be formulated as a
joint block problem.

That is, given a sub-block cube

- `D_0, D_1, ..., D_{N-1}`

we define a likelihood for the full block and optimize or evaluate it jointly
over the active parameters for that block.

This is more naturally described as a block fit or smoothing-style inference
than as independent per-frame fits.

### Across sub-blocks

After one sub-block is processed, the resulting parameter estimates and
uncertainty should inform the next sub-block.

Conceptually, this is the recursive or filtering layer of the mission
estimation story:

- propagate the belief state forward to the next sub-block
- use the next sub-block’s image data to update that belief
- continue over time

This document does not require that the first implementation be a full Bayesian
filter. The important design point is that sub-block inference should be
compatible with that future interpretation.

## Problem formulation

Let the observed sub-block be a stack of frames

- `D_0, D_1, ..., D_{N-1}`

Let the active latent state be partitioned into:

- shared block parameters `theta_shared`
- frame-varying parameters `phi_i`

For the first phase, we take:

- `theta_shared = {}` as an empty inferred shared set
- `phi_i = [x_i, y_i, pa_i]`

where:

- `x_i = source.x_position_as`
- `y_i = source.y_position_as`
- `pa_i = source.position_angle_deg`

The model prediction for frame `i` is produced by:

1. starting from the shared base store
2. applying any fixed shared assumptions
3. applying the frame-specific active overrides
4. refreshing derived values
5. evaluating the forward model for that frame

The full block objective should be understood conceptually as:

- `L_block = data_term + prior_term + temporal_term`

where:

- `data_term` is the image-domain likelihood contribution over the full cube
- `prior_term` contains active priors on shared and/or frame parameters
- `temporal_term` contains any explicit temporal regularization or motion model

For the first implementation, this simplifies to a block image loss with only
frame-varying registration parameters active. The broader form is recorded here
so the early schema does not point us in the wrong direction.

## State taxonomy

A useful working taxonomy is:

### Structural state

Parameters that define the model structure and are not intended to vary during
routine sub-block inference.

Examples:

- detector layer topology
- pupil grids or structural optical configuration
- model kind selections

### Slowly varying calibration state

Parameters that may change over long spans but are not expected to vary within a
single 1-second sub-block.

Examples:

- detector calibration maps
- optical alignment terms
- selected static or quasi-static aberrations

These may often live in the belief state while remaining frozen during routine
science solves.

### Shared sub-block state

Parameters assumed constant across one sub-block but potentially inferable.

Examples:

- separation
- contrast or flux normalization
- plate scale
- selected low-order Zernikes

These are strong candidates for the next inference phase after registration-only
recovery.

### Frame-varying state

Parameters that vary within the sub-block.

Examples:

- `source.x_position_as`
- `source.y_position_as`
- `source.position_angle_deg`

This is the first active inference set.

## First implementation milestone

The first implementation remains intentionally narrow.

### Active inference state

Infer per-frame registration parameters for every frame in the block:

- `source.x_position_as`
- `source.y_position_as`
- `source.position_angle_deg`

### Frozen state

Hold all shared system/source/optics/detector parameters fixed across the block,
including for example:

- separation
- contrast
- total flux
- plate scale
- optical aberration state
- detector calibration state

### Why this first milestone is still useful

This first phase is still the right place to start because it:

- exercises the multi-frame block-likelihood path
- validates artifact and data-flow assumptions
- makes the frame-varying state explicit
- provides interpretable truth-vs-recovered diagnostics
- gives a clean foundation for later joint shared-plus-frame inference

The important change in this note is not the milestone itself, but the way that
milestone is framed relative to the longer-term design.

## Initialization

Initialization should be treated as conceptually separate from priors.

For the first implementation:

- initialize every frame from the same shared starting values for `x`, `y`, and
  `PA`, unless the config explicitly provides something else
- keep initialization simple and explicit
- do not over-engineer frame-specific initialization strategies yet

Longer term, initialization may come from:

- a propagated belief state from the previous sub-block
- a centroid or coarse registration estimate
- the previous block’s posterior mean
- a shared-plus-residual temporal model

The initial recipe does not need to support all of these, but the design should
leave room for them.

## Priors and uncertainty

The long-term design should assume that nearly every important modeled quantity
can have uncertainty attached to it in the belief state.

However, the active use of that uncertainty depends on whether the quantity is
currently part of the active inference state.

### For active parameters

Priors should be supported eventually for:

- shared inferred parameters
- frame-varying inferred parameters
- temporal-model hyperparameters when present

These priors may be weak or strong depending on operational knowledge.

### For frozen parameters

Frozen parameters may still carry uncertainty in the mission belief state, but
that uncertainty does not automatically enter the local sub-block solve.

To make frozen uncertainty matter in the local solve, we would need one of:

- joint inference
- marginalization
- approximate uncertainty propagation / nuisance-state treatment

That distinction should be kept clear in both code and docs.

## Loss and objective structure

The first recipe should remain readable and recipe-like, but the design target
should be broader than a hard-coded registration-only NLL script.

The intended conceptual structure is:

1. load and resolve config
2. load the sub-block cube and associated metadata
3. build the forward spec and shared base store
4. construct the active state for the current inference phase
5. define the block objective:
   - image-domain likelihood
   - optional active priors
   - optional temporal regularization or motion model
6. initialize the active parameters
7. run the optimizer or estimator
8. write outputs and diagnostics
9. summarize the updated state estimate for downstream use

The first implementation may only instantiate a subset of this structure, but
the design note should describe the full intended pattern.

## Inputs

Expected operational inputs remain:

- an observation sub-block FITS cube
- a canonical config / prescription describing the shared fixed or partially
  inferred model state and experiment settings
- optional frame-truth CSV for evaluation and comparison
- optional manifest JSON for metadata and artifact discovery

The common synthetic-data workflow should still be:

- point inference at the rendered cube
- auto-discover `manifest.json` beside that cube
- infer the truth-trace path from the render manifest when available

But the design should not assume that truth artifacts will exist in real
mission operations. Truth is a simulation and validation convenience, not a
mission-mode requirement.

## Output products

Outputs should support both immediate algorithm assessment and future recursive
use.

Recommended outputs include:

- a run manifest / results summary
- a recovered per-frame parameter table
- a comparison table with truth and recovered values when truth is available
- residual or fit-quality diagnostics
- simple plots that compare truth vs recovered traces
- a machine-readable summary of the recovered active state suitable for later
  propagation into a running belief state

At minimum, the outputs should make it easy to answer:

- did the inference recover the intended motion pattern?
- where does recovery succeed or struggle?
- how do the residuals behave across the block?
- what state estimate should be carried forward?

The run manifest should capture enough shared-state context to review the fit
without reopening the original prescription, including:

- source `config_path`
- resolved input cube/trace/manifest paths
- whether the render manifest was auto-discovered
- resolved fixed or assumed `system` config snapshot
- active initialization and optimizer settings
- objective settings
- summary metrics
- recovered-state artifact paths

## Recommended diagnostics

The first implementation should favor a few highly interpretable diagnostics over
a large analysis suite.

Useful first diagnostics include:

- truth vs recovered `x(t)`
- truth vs recovered `y(t)`
- truth vs recovered `PA(t)`
- residual traces for each recovered parameter
- one or more image residual summaries across the block

Later diagnostics may also include:

- shared-parameter posterior summaries
- temporal-model residual checks
- block-to-block state continuity summaries

## What is intentionally out of scope in the first implementation

The first implementation should not attempt to solve the entire mission problem.

Out of scope for the first implementation:

- inference of arbitrary shared parameter sets
- full recursive filtering across blocks
- generalized temporal-model inference
- calibration-map inference inside the routine science solve
- multi-ROI joint inference
- full uncertainty propagation from all frozen nuisance parameters
- elaborate abstraction layers that obscure the recipe logic

These are deferred not because they are unimportant, but because the first goal
is to establish a clear and trustworthy block-inference pattern.

## Planned next phases

### Next phase: shared-plus-frame inference

After registration-only recovery, the next phase should allow a small shared
parameter set to be inferred jointly with frame-varying registration.

The first shared-parameter target should likely be:

- `optics.plate_scale_as_per_pix`

This is a natural next step because plate scale is a shared quantity that may
not be perfectly known, and small errors in plate scale can couple directly into
registration recovery.

### After that: temporal structure for frame state

A later phase should introduce an explicit temporal model for frame-varying
registration, for example:

- independent frame parameters
- shared anchor plus residuals
- linear drift plus jitter
- random walk

This would move the block estimator closer to a mission-like smoothing problem.

### Later still: recursive block-to-block state update

Once sub-block estimation is working cleanly, the next architectural step is to
define how one sub-block’s recovered state informs the next sub-block.

This is the stage at which a more explicit recursive filter or belief-propagation
story becomes appropriate.

## Design guidance for implementation

When implementing the first recipe and its early extensions, the main priorities
should be:

- readability
- alignment with canonical recipe structure
- explicit control flow
- minimal abstraction
- outputs that are easy to interpret
- schema choices that do not contradict the intended long-term architecture

In practice, that means:

- keep the main recipe logic visible
- use helper functions sparingly and only where they genuinely clarify the
  script
- avoid naming that hard-codes synthetic “truth” semantics into the long-term
  inference story
- keep initialization, priors, and fixed assumptions conceptually separate
- make it easy to inspect recovered values, residuals, and carried-forward state

## Success criteria

This design direction will be successful if we build a workflow that:

- loads a sub-block cube and fixed or partially inferred model state
- works cleanly when only the cube path is provided in the standard artifact
  layout
- supports a clear first milestone of jointly inferring per-frame `x/y/PA`
  across the full block
- produces readable outputs comparing recovered traces to truth in synthetic
  studies
- naturally extends to shared-plus-frame inference
- naturally extends to block-to-block state propagation
- provides a clear foundation for a future mission-aligned estimation story

## Summary

The first observation sub-block inference recipe is still intentionally modest:
a registration-only block fit over a short image cube.

But that recipe should be understood as the first restricted case of a broader
hierarchical time-domain estimation problem.

The intended long-term direction is:

- batch or smoothing-style inference within each short fixed-ROI sub-block
- recursive propagation of a broader belief state across successive sub-blocks
- eventual support for both shared and frame-varying inferred parameters
- clear separation between belief state, active parameters, frozen parameters,
  priors, initialization, and temporal structure

That framing should help keep the implementation, schema, and future extensions
pointed in the right direction.