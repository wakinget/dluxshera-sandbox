# Observation Sub-Block Inference Design Note

## Purpose

This note defines the intended direction for observation sub-block inference.

The central design goal is to treat sub-block inference as a structured
time-domain estimation problem rather than as a one-off optimization over a
small synthetic dataset. This will help explain how we intend to support
mission-level inference.

The immediate implementation goal remains a readable block-inference recipe
that jointly solves for frame-varying registration parameters over a short stack
of frames. However, this document is intentionally broader than this first
milestone. It describes the longer-term mission-aligned inference story so that
the initial script, schema, and artifacts grow in the right direction.

## Context

We currently have a working sub-block workflow with three main pieces:

- a trace-generation stage that produces truth tables for a specified sub-block
- a rendering stage that produces a sub-block FITS cube from a specified 
  trace/truth table
- an inference stage that begins by fitting per-frame registration terms from
  the rendered cube. This stage is expected to change and grow over time as needed 
  to include shared terms, and/or assumptions about how parameters evolve over time.

The first prototype is intentionally narrow: it will fit only per-frame registration 
terms, holding all other system parameters fixed. This is a useful first milestone,
but it is not the full mission inference story.

A more realistic mission-level view is:

- the instrument carries a time-evolving belief about many/all parameters
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

The working assumption is:

- frame cadence is approximately 20 Hz
- the ROI remains fixed for roughly 1 second
- a new ROI is chosen approximately every 1 second
- each fixed-ROI interval therefore contains about 20 frames

This naturally defines an observation sub-block as the interval over which one
ROI crop remains fixed. The 1-second interval is assumed for now, but may be
changed in the future.

That structure motivates a hybrid estimation strategy:

- **within one sub-block:** use a joint block fit over all frames in the block
- **across successive sub-blocks:** propagate a running belief state forward in
  time, updating it as each new block arrives

## Design goal

The long-term goal is to perform inference on a sequence of observation
sub-blocks using:

- a forward image model
- a partition between shared and frame-varying latent state
- priors or state uncertainty for all important modeled quantities
- temporal structure for the frame-varying state
- a mechanism for carrying information from one sub-block to the next

The first recipe script will not implement all of this, but it should be
understood as the first restricted case of that broader problem.

## Terminology

To avoid conflating several different ideas, this note uses the following terms.

### Belief state

The mission’s current estimate of (all) parameter values and uncertainty.

This may include nearly every important modeled quantity, even when a given
parameter is not actively varied in the current solve. Examples include:

- binary separation
- photometry terms
- system plate scale
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

In this note, we distinguish between "prior" and "belief state." A prior
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
problem becomes more realistic. We will want to take advantage of the knowledge
that certain parameters may be correlated across frames.

## Problem formulation

Let the observed sub-block be a stack of frames:

- `D_0, D_1, ..., D_{N-1}`

Let the active latent state be partitioned into:

- shared block parameters `theta_shared`
- frame-varying parameters `phi_i`

For our first phase, we take:

- `theta_shared = {}` as an empty inferred shared set
- `phi_i = [x_i, y_i, pa_i]` as the per-frame registration terms

where:

- `x_i = source.x_position_as`
- `y_i = source.y_position_as`
- `pa_i = source.position_angle_deg`

The full block objective (Loss function) can be understood conceptually as:

- `L_block = data_term + prior_term + temporal_term`

where:

- `data_term` is the image-domain likelihood contribution over the full cube (NLL summed over cube)
- `prior_term` contains active priors on shared and/or frame parameters
- `temporal_term` contains any explicit temporal regularization or motion model

For the first implementation, this simplifies to a block image loss with only
frame-varying registration parameters active.

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
single sub-block.

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

Mainly:

- `source.x_position_as`
- `source.y_position_as`
- `source.position_angle_deg`

This is the first active inference set.

## Hierarchical estimation levels

The intended architecture is hierarchical.

### Level 1: frame level

This is the raw image domain.

Data product:

- individual short-exposure frames within one sub-block

Typical varying quantities:

- frame-to-frame registration

Role:

- provides the image-domain likelihood
- is not, by itself, the main carried-forward product

### Level 2: sub-block level

This is the first routine inference unit.

Data product:

- one fixed-ROI frame cube
- associated metadata and any optional truth products in simulation mode

Typical active quantities:

- per-frame `x/y/PA`
- a small set of shared parameters such as plate scale

Role:

- performs one joint block fit
- outputs recovered active state, diagnostics, and a machine-readable state
  summary for later propagation
- The state summary is what we carry forward to the next level, but we don't
  discard the data cubes.

### Level 3: observation-period level

This is the layer that links many sub-blocks from one observing window.

Data product:

- a sequence of recovered sub-block state summaries
- optional access to retained image cubes when re-analysis is needed

Typical quantities of interest:

- slowly drifting shared parameters
- calibration terms that are stable over many blocks
- observation-level astrometric summaries

Role:

- propagates belief state from one sub-block to the next
- accumulates information across many local solves
- may occasionally revisit the image-domain products when a richer model is
  needed

### Level 4: mission level

This is the longest-timescale layer.

Data product:

- a sequence of observation-level summaries across many visits or epochs

Typical quantities of interest:

- long-term calibration state
- long-term drift terms
- final science parameters and their uncertainty

Role:

- fuses information over mission timescales
- updates long-lived priors and calibration beliefs
- supports mission-scale performance assessment and science analysis

## How sub-block inference fits into the hierarchy

A sub-block solve should be thought of as producing a local state update.

In routine operation, the carried-forward product should usually be a recovered
state summary rather than a composite image. Coadded or aligned images may still
be useful as diagnostics, but they are not the primary abstraction for the
higher-level estimator.

A useful way to phrase the transition is:

- images are the evidence
- sub-block inference is the local estimator
- the recovered state summary is the propagated product

## What one sub-block solve should output

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
- resolved fixed or assumed `system` config snapshot
- active initialization and optimizer settings
- objective settings
- summary metrics
- recovered-state artifact paths

## Example mechanics for hierarchical estimation

The exact machinery is not finalized, but the intended mechanics are roughly as
follows.

### Example A: first milestone

For one sub-block:

1. start from an assumed shared state
2. fit per-frame `x/y/PA` jointly across the full block
3. write recovered per-frame traces and fit diagnostics
4. summarize the recovered block state
5. pass a simple summary forward for use as the next block’s initialization

In this first milestone, the shared state is fixed. The value of this phase is
that it validates the block likelihood, the artifact layout, and the basic
frame-varying estimation path.

### Example B: shared-plus-frame block inference

For one sub-block:

1. start from an assumed shared state plus uncertainty
2. fit per-frame `x/y/PA`
3. also fit one or a few shared terms, for example:
   - `optics.plate_scale_as_per_pix`
4. write recovered frame traces and recovered shared terms
5. carry the recovered shared estimate and uncertainty into the next block

This is the natural next phase after registration-only recovery.

### Example C: block-to-block propagation over one observing period

For a sequence of sub-blocks:

1. initialize block `k` from the propagated state summary from block `k-1`
2. solve block `k` in the image domain
3. update the belief state using the recovered block summary
4. continue to the next block

In this picture, the routine estimator advances primarily on sub-block summaries,
not by re-fitting all previous frames from scratch after every update.

### Example D: higher-level refinement

At some later stage, a higher-level solve may revisit a set of retained sub-block
cubes together when a richer parameter set becomes important.

For example:

- a first pass may recover only registration
- a second pass over selected blocks may promote plate scale or selected
  low-order Zernikes into the active state
- a still later pass may update long-timescale calibration parameters using many
  blocks together

This means sub-block summaries are the main propagated product, but not a
lossless replacement for the underlying images.

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

### For frozen parameters

Frozen parameters may still be uncertain in the broader belief state.

In many early solves, that uncertainty will be ignored locally for simplicity.
Later, it may need to be handled by one or more of:

- promoting the parameter into the active state
- marginalizing approximately over its uncertainty
- carrying uncertainty inflation into higher-level summaries
- re-solving selected blocks with a richer active model

The first implementation does not need to solve this fully. It only needs to
avoid language that falsely implies frozen state is certain.

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

- independent frame parameters (first assumption)
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
are:

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
- avoid naming that hard-codes synthetic truth semantics into the long-term
  inference story (we won't always know what truth is).
- keep initialization, priors, and fixed assumptions conceptually separate
- make it easy to inspect recovered values, residuals, and carried-forward state

## Success criteria

This design direction will be successful if we build a workflow that:

- loads a sub-block cube and fixed or partially inferred model state
- supports a clear first milestone of jointly inferring per-frame `x/y/PA`
  across the full block
- produces readable outputs comparing recovered traces to truth in synthetic
  studies
- naturally extends to shared-plus-frame inference
- naturally extends to block-to-block state propagation
- makes the carried-forward state explicit and inspectable
- provides a clear foundation for a future mission-aligned estimation story

## Summary

The first observation sub-block inference recipe is still intentionally modest:
a registration-only block fit over a short image cube.

The intended long-term direction is:

- joint inference within each short fixed-ROI sub-block
- propagation of a broader belief state across successive sub-blocks
- eventual support for both shared and frame-varying inferred parameters
- clear separation between belief state, active parameters, frozen parameters,
  priors, initialization, temporal structure, and carried-forward state

That framing should help keep the implementation, schema, and future extensions
pointed in the right direction.