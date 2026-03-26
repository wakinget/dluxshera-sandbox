# Observation Sub-Block Inference Design Note

## Purpose

This note defines the first multi-frame inference problem for the observation sub-block workflow.

The immediate goal is to move from sub-block generation and visualization to a readable, recipe-style inference script that operates on a short stack of frames and jointly solves for the frame-varying registration parameters across the block.

This first inference phase is intentionally narrow. It is meant to establish the basic block-likelihood workflow, artifact handling, and result interpretation before we expand to shared-parameter inference.

## Context

We now have a working sub-block workflow with three main pieces:

- a renderer that generates an observation sub-block FITS cube from a shared base state plus per-frame registration traces
- a trace-builder that generates canonical explicit trace CSV inputs
- a visualization utility that helps inspect the resulting cube, traces, and summary diagnostics

The next step is to define the corresponding inference problem.

The longer-term direction is a hierarchical time-domain inference story in which:

- some parameters are shared across a sub-block
- some parameters vary frame-to-frame
- later phases may allow block inference to inform imperfectly known shared parameters
- still later phases may incorporate slowly drifting shared parameters across longer spans of data

This note only defines the first block-inference milestone.

## Scope of First Phase

The first phase focuses on **registration-only block inference**.

The first block-inference recipe should:

- take an observation sub-block image cube as input
- treat the shared system/source/optics/detector state as fixed
- jointly infer the per-frame registration parameters for every frame in the block:
  - `source.x_position_as`
  - `source.y_position_as`
  - `source.position_angle_deg`

This phase is intended to answer a simple but important question:

Can we recover the injected frame-varying registration traces by fitting the whole sub-block jointly, while holding the rest of the model fixed?

## Why

This is the simplest meaningful multi-frame inference problem supported by the current sub-block workflow.

It is a good first step because:

- it directly exercises the observation sub-block data model
- it keeps the parameter partition conceptually clear
- it avoids immediately mixing frame-varying and shared-parameter uncertainty
- it is easy to visualize against truth traces
- it should fit naturally into the current canonical recipe style

This phase is primarily about making the block inference mechanics clear and readable.

## Problem formulation

Let the observed sub-block be a stack of frames

- `D_0, D_1, ..., D_{N-1}`

Let the fixed shared parameter state for the block be represented by a single base forward-model store.

For each frame `i`, define a frame-specific parameter vector

- `phi_i = [x_i, y_i, pa_i]`

where:

- `x_i = source.x_position_as`
- `y_i = source.y_position_as`
- `pa_i = source.position_angle_deg`

The model prediction for frame `i` is produced by:

1. starting from the shared base store
2. applying the frame-specific overrides for `x_i`, `y_i`, and `pa_i`
3. refreshing derived values
4. evaluating the forward model for that frame

The total block loss is the sum of per-frame losses over the cube.

Conceptually:

- `L_block = sum_i L_i(phi_i ; D_i, shared_base_state)`

For this first phase, there are no inferred shared parameters. The only optimized variables are the per-frame registration terms.

## Inputs

The first block-inference recipe should be designed around the current observation sub-block artifacts.

Expected inputs:

- an observation sub-block FITS cube
- a canonical config / prescription describing the fixed shared model state and experiment settings
- optional frame-truth CSV for evaluation and comparison
- optional manifest JSON for metadata and artifact discovery

The key operational input is the image cube itself. Truth and manifest artifacts are primarily useful for diagnostics, comparison, and experiment bookkeeping.

## Parameter partition

### Inferred

Per-frame registration parameters for each frame:

- `source.x_position_as`
- `source.y_position_as`
- `source.position_angle_deg`

### Fixed

Everything else is held fixed for the full sub-block, including for example:

- binary separation
- contrast
- total flux
- plate scale
- optics terms
- detector calibration state

This fixed-vs-varying split is deliberate. It keeps the first implementation readable and makes the block-likelihood structure easy to understand.

## Initialization

The initial guess for the per-frame registration parameters should come from the current experiment/config path in a way that is easy to read in the recipe.

Reasonable initial behavior for the first implementation:

- initialize every frame from the same shared starting values for `x`, `y`, and `PA`, unless the config explicitly provides something else
- keep initialization simple and explicit
- do not over-engineer frame-specific initialization strategies yet

This phase is about the block-loss plumbing, not about sophisticated initialization.

## Loss structure

The first recipe should follow the canonical recipe style as closely as possible.

The intended flow is:

1. load and resolve config
2. load the sub-block cube and associated metadata
3. build the forward spec and shared base store
4. define the frame-varying infer keys
5. build a block loss that loops over frames and sums per-frame losses
6. initialize the optimization variables
7. run the optimizer
8. write results and diagnostics

The recipe should remain easy to read top-to-bottom. Small helper functions are fine where they genuinely improve readability, but the main control flow should stay visible in the recipe.

## Output products

The first block-inference recipe should write outputs that make recovery quality easy to interpret.

Recommended outputs:

- a run manifest / results summary
- a recovered per-frame parameter table
- a comparison table with truth and recovered values when truth is available
- residual or fit-quality diagnostics
- simple plots that compare truth vs recovered traces

At minimum, the outputs should make it easy to answer:

- did the inference recover the injected registration motion?
- where does recovery succeed or struggle?
- how do the residuals behave across the block?

## Recommended diagnostics

The first implementation should favor a few highly interpretable diagnostics over a large analysis suite.

Useful first diagnostics include:

- truth vs recovered `x(t)`
- truth vs recovered `y(t)`
- truth vs recovered `PA(t)`
- residual traces for each recovered parameter
- one or more image residual summaries across the block

These should be sufficient for the first milestone.

## What is intentionally out of scope in this first phase

This first block-inference phase should not try to solve everything at once.

Out of scope:

- inference of shared parameters
- slow drifts in shared parameters across the block
- arbitrary frame-varying parameter sets
- multi-ROI or downlink-style observation products
- a generalized time-series inference framework
- elaborate validation policy beyond what is needed for the current recipe
- extensive abstraction layers that obscure the recipe logic

The goal is a readable first block-inference recipe, not a fully general inference engine.

## Planned next step

The next phase after registration-only block inference should allow a small set of shared parameters to be inferred jointly with the frame-varying registration terms.

The first shared-parameter target should likely be:

- `optics.plate_scale_as_per_pix`

This next phase is motivated by a realistic use case:

- the assumed shared parameter state may not be perfectly correct
- the generated data may reflect a slightly different true shared state
- given enough frame information, the block inference may be able to correct small errors in shared parameters while still solving for per-frame registration

That is the next important demonstration after the first phase.

Conceptually, the second phase would move from:

- registration-only block inference

to:

- joint block inference with
  - frame-varying `x/y/PA`
  - a small shared parameter set, starting with plate scale

## Design guidance for implementation

When we implement the first recipe, the main priorities should be:

- readability
- alignment with canonical recipe structure
- explicit control flow
- minimal abstraction
- outputs that are easy to interpret

In practice, that means:

- keep the main recipe logic visible
- use helper functions sparingly and only where they genuinely clarify the script
- prefer straightforward packing/unpacking over clever hidden machinery
- keep the first optimizer path simple
- make it easy to inspect truth, recovered values, and residuals

## Success criteria

This will be successful if we have a recipe that:

- loads a sub-block cube and fixed model state
- jointly infers per-frame `x/y/PA` across the full block
- produces readable outputs comparing recovered traces to truth
- fits naturally into the current recipe-oriented workflow
- provides a clear foundation for the next phase of shared-plus-frame inference

## Summary

This first inference milestone is intentionally modest.

It is not yet about solving the full science problem. It is about establishing a clean and readable block-inference pattern that matches the current sub-block generation workflow and gives us confidence in the mechanics of multi-frame fitting.

Once this registration-only block inference is working, the natural next step is to let the block jointly inform a small set of shared parameters, beginning with plate scale.