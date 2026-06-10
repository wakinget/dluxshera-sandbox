# Trajectory-Driven Observation Campaign Design

## Implemented first path

The first implemented trajectory-driven path is centered on:

- `src/dluxshera/utils/obs_subblock_trajectory.py`
- `examples/scripts/run_trajectory_subblock_campaign.py`
- the Airbus example CSV at
  `src/dluxshera/data/airbus_data/Thirty_Min_Observation_Window.csv`

The Airbus adapter reads X, Y, and Z pointing samples. X and Y map directly to
`source.x_position_as` and `source.y_position_as` in arcseconds. Z maps to
`source.position_angle_deg` after conversion from arcseconds to degrees with
`deg = arcsec / 3600`.

The current preparation workflow:

1. loads the raw Airbus trajectory
2. normalizes it to canonical dLuxShera frame keys
3. selects a requested time window
4. linearly interpolates the trajectory to the frame cadence, normally 50 ms
5. splits the selected frames into non-overlapping subblocks
6. writes each subblock's `frame_truth.csv`
7. fits a per-subblock line for each active registration key and writes
   `starting_guess_prediction.csv`
8. writes case-local trace/render/inference config records and `command.sh`

For a 20-frame, 50 ms subblock, frame times are
`t_block_start + i * 0.05` for `i = 0 ... 19`, so one block covers 0.00 through
0.95 s and the next block starts at 1.00 s. The exact boundary frame is not
duplicated.

The renderer still consumes the canonical explicit trace schema:

```text
frame_index,time_s,source.x_position_as,source.y_position_as,source.position_angle_deg
```

Single-star calibration-style preparation can omit PA by passing explicit
output keys, for example:

```bash
PYTHONPATH=src python examples/scripts/run_trajectory_subblock_campaign.py \
  --run-name airbus_single_star_xy_dryrun \
  --trajectory-csv src/dluxshera/data/airbus_data/Thirty_Min_Observation_Window.csv \
  --duration-s 2.0 \
  --source-kind single_star \
  --output-keys source.x_position_as,source.y_position_as \
  --dry-run
```

The recovered-reference inference recipe now supports
`experiment.inference.init.frame.mode: starting_guess_csv`. The config maps
active frame keys to columns in `starting_guess_prediction.csv`, for example:

```yaml
inference:
  init:
    frame:
      mode: starting_guess_csv
      path: starting_guess_prediction.csv
      columns:
        source.x_position_as: source.x_position_as_linear_fit
        source.y_position_as: source.y_position_as_linear_fit
        source.position_angle_deg: source.position_angle_deg_linear_fit
```

This starting-guess CSV is optimizer initialization only. It is not truth.

## Example commands

Dry-run two seconds of Airbus trajectory preparation:

```bash
PYTHONPATH=src python examples/scripts/run_trajectory_subblock_campaign.py \
  --run-name airbus_trajectory_dryrun_2s \
  --trajectory-csv src/dluxshera/data/airbus_data/Thirty_Min_Observation_Window.csv \
  --start-s 0.0 \
  --duration-s 2.0 \
  --frame-dt-s 0.05 \
  --n-frames-per-subblock 20 \
  --source-kind binary \
  --phi-ref recovered \
  --dry-run
```

Prepare and execute a small two-block smoke through the existing study runner:

```bash
PYTHONPATH=src python examples/scripts/run_trajectory_subblock_campaign.py \
  --run-name airbus_trajectory_smoke_2blocks \
  --trajectory-csv src/dluxshera/data/airbus_data/Thirty_Min_Observation_Window.csv \
  --start-s 60.0 \
  --duration-s 2.0 \
  --frame-dt-s 0.05 \
  --n-frames-per-subblock 20 \
  --source-kind binary \
  --phi-ref recovered \
  --max-workers 1 \
  --run-children
```

The generated layout is:

```text
Results/trajectory_subblock_campaign/<run_name>/
  campaign_plan.json
  resolved_config.json
  trajectory_ingest_summary.json
  subblock_plan.csv
  subblocks/
    subblock_000000/
      frame_truth.csv
      starting_guess_prediction.csv
      trace_config.json
      render_config.json
      inference_config.json
      command.sh
```

Current limitations remain explicit:

- dynamic cropping is not implemented
- `psf_npixels` / ROI-origin realism is not tested by this path
- high-order WFE map insertion is not implemented
- the trajectory currently affects frame-level source registration truth and
  starting guesses only

## Campaign Wrapper Integration

The main campaign wrappers also support trajectory trace sources through
`subblocks.trace_source.mode`:

- `iid_jitter` preserves the legacy trace-template behavior and remains the
  default
- `trajectory` materializes Airbus-derived `frame_truth.csv` and
  `starting_guess_prediction.csv` during campaign planning
- `external_plan` reuses a previously prepared trajectory campaign
  `campaign_plan.json` plus `subblock_plan.csv`

Integrated wrappers:

- `examples/scripts/run_single_star_calibration_demo.py`
- `examples/scripts/run_observation_bias_campaign.py`

Single-star trajectory mode defaults to X/Y active/output keys. Binary
observation-bias mode defaults to X/Y/PA. In trajectory and external-plan modes,
child commands pass:

```text
--external-frame-truth-csv <frame_truth.csv>
--starting-guess-csv <starting_guess_prediction.csv>
--starting-guess-mode starting_guess_csv
```

Aggregate-only flows load stored campaign/subblock plans and should not
reinterpret a new trajectory configuration. Resume reuses already materialized
trajectory artifacts and fails clearly if required files are missing.

## Purpose

This note defines a near-term design for using a realistic pointing trajectory to
construct observation-scale synthetic data campaigns for sub-block inference
experiments.

The immediate motivation is a sample X/Y pointing trajectory provided by the
presumed satellite bus provider. The trajectory spans roughly one 30-minute
observation window and can be used as a realistic motion baseline for generating
many observation sub-blocks. Those sub-blocks can then exercise the current
trace, render, and inference workflow in a way that is closer to the eventual
mission algorithm demonstration than isolated toy sub-blocks.

This document is intentionally scoped between two existing layers:

- below mission-scale astrometric simulations that span years and may include
  planet/no-planet truth models
- above the existing per-sub-block explicit trace, rendering, and inference
  recipes

The goal is to introduce a durable campaign-generation layer without prematurely
building the full observation-level estimator or mission-level detection
pipeline.

## Background and motivation

The current observation sub-block workflow is built around a useful explicit
boundary:

1. generate or provide per-frame truth traces
2. render an image cube for one sub-block
3. run inference on that cube
4. write diagnostics and recovered trace artifacts

That remains the right low-level contract. However, proposal-facing algorithm
demonstrations need to show more than one isolated sub-block. We need to show
how the algorithm behaves across a realistic observation-like time sequence:

- pointing varies throughout the observation
- noise should average down as more data are accumulated
- model mismatch may create coherent biases rather than independent errors
- sub-block products need to reduce into observation-level summaries
- diagnostics should help explain when the algorithm works and when it fails

A provider trajectory gives us a practical reference input for this work. It is
not necessarily a formal pointing requirement baseline, but it is realistic
enough to support early algorithm development and proposal narrative plots.

## Design goals

The trajectory-driven campaign layer should:

- ingest a long-timescale pointing trajectory from an external file
- preserve provenance and units/conventions for the trajectory
- select a manageable observation window, initially a subset such as the first
  24--60 seconds
- resample or interpolate the trajectory onto frame centers when needed
- partition the frame sequence into one-second observation sub-blocks
- write one explicit frame-truth CSV per sub-block, compatible with the existing
  renderer
- optionally generate renderer and inference prescription stubs for each
  sub-block
- write a campaign-level manifest describing inputs, configuration, generated
  sub-blocks, and intended follow-on commands
- support later aggregation of sub-block inference outputs into observation-level
  diagnostics

The first implementation should emphasize trace/campaign organization and
provenance, not full estimator sophistication.

## Non-goals for the first implementation

The first implementation should not attempt to:

- simulate all 30 minutes by default
- render all 1800 one-second sub-blocks by default
- implement a full Bayesian observation-level filter
- solve the mission-scale planet-detection problem
- infer shared parameters across many sub-blocks
- decide the final jitter/smear physical model
- replace the existing explicit frame-truth CSV renderer contract
- require the estimability/FIM track to be complete first

Those topics are important, but they should remain separate design and
implementation tracks.

## Relationship to existing sub-block workflow

The current workflow can be summarized as:

```text
one sub-block trace CSV
  -> one rendered FITS cube
  -> one sub-block inference run
  -> one set of diagnostics and recovered state products
```

The trajectory-driven workflow should wrap that without breaking the existing
contract:

```text
long provider trajectory
  -> selected observation window
  -> many per-sub-block trace CSVs
  -> many rendered FITS cubes
  -> many sub-block inference runs
  -> observation-level summaries and proposal plots
```

The existing renderer should continue to consume explicit per-frame trace CSVs.
The new campaign layer should therefore be responsible for translating a
higher-level trajectory file into those explicit sub-block traces.

## Time-scale hierarchy

This work introduces an explicit distinction between several time layers.

### Mission truth layer

This is the longest-timescale truth description. It may eventually contain a
multi-year target astrometric model, including binary motion with or without an
exoplanet perturbation.

Examples:

- true binary separation as a function of mission time
- true binary position angle as a function of mission time
- planet/no-planet scenario labels
- long-timescale instrument drift models

This layer is out of scope for the first trajectory-driven campaign task, but
the campaign schema should not prevent adding it later.

### Observation trajectory layer

This layer describes motion over one observation window, roughly 30 minutes in
the current concept.

Examples:

- bus/provider X pointing trajectory
- bus/provider Y pointing trajectory
- optional roll/PA trajectory if available
- optional quality flags or metadata

This is the main new input for the first implementation.

### Sub-block layer

This layer partitions an observation window into short intervals, likely one
second each in the current working assumption.

Examples:

- subblock_000000 spans 0--1 s
- subblock_000001 spans 1--2 s
- subblock_000059 spans 59--60 s

Each sub-block should have its own trace, render, and inference folders so that
individual blocks can be run, debugged, or regenerated independently.

### Frame layer

This is the existing explicit truth layer used by the renderer.

Examples:

- frame cadence: 20 Hz
- exposure time: 50 ms
- frame-level `source.x_position_as`
- frame-level `source.y_position_as`
- frame-level `source.position_angle_deg`, if driven or synthesized

The campaign generator should output this layer as explicit per-frame CSVs.

## Proposed repository layout

The first trajectory file should initially be treated as example data rather
than core package data, unless licensing/provenance or repeated usage makes it a
canonical package asset later.

Recommended input layout:

```text
examples/data/trajectories/
  provider_sample_pointing_30min.csv
  provider_sample_pointing_30min_metadata.yaml
  README.md
```

Recommended generated campaign layout:

```text
Results/trajectory_campaigns/<campaign_id>/
  manifest.json
  campaign_prescription.yaml
  trajectory/
    normalized_trajectory.csv
    trajectory_metadata.json
  subblocks/
    subblock_000000/
      trace/
      render/
      inference/
    subblock_000001/
      trace/
      render/
      inference/
    ...
  aggregate/
    # later: observation-level summaries and plots
```

This mirrors the existing trace/render/inference mental model while adding a
campaign-level root for provenance and aggregation.

## Input trajectory contract

The exact provider file format may vary. The first implementation should define
a normalized internal trajectory table rather than binding all downstream code to
one raw file format.

A normalized trajectory should contain at least:

```text
time_s
x_pointing_as
y_pointing_as
```

Optional columns may include:

```text
roll_deg
quality_flag
source_row_index
```

Open questions that must be resolved before using the data quantitatively:

- Are X/Y values angular offsets, focal-plane offsets, detector pixels, or some
  other coordinate?
- What are the units?
- What is the sign convention?
- Are values in an inertial, body, telescope, focal-plane, or detector frame?
- Are the samples instantaneous, averaged over an interval, or command/estimate
  values?
- Does the trajectory already include jitter, or only lower-frequency pointing?
- Is the time base relative to observation start, absolute mission time, or
  another epoch?

These should be recorded in the trajectory metadata sidecar before the file is
used for proposal-grade plots.

## Campaign prescription concept

A trajectory-driven campaign should be configurable from a single prescription
file. The exact schema can evolve, but the first version could look like:

```yaml
experiment:
  kind: trajectory_observation_campaign
  notes: "First 60 seconds of provider pointing trajectory for sub-block demo."
  seed: 42

  trajectory_campaign:
    trajectory:
      path: ../../examples/data/trajectories/provider_sample_pointing_30min.csv
      metadata_path: ../../examples/data/trajectories/provider_sample_pointing_30min_metadata.yaml
      format: csv
      time_column: time_s
      x_column: x_pointing_as
      y_column: y_pointing_as
      units:
        x: arcsec
        y: arcsec
      coordinate_convention: TBD

    window:
      start_time_s: 0.0
      duration_s: 60.0

    frames:
      cadence_hz: 20.0
      exposure_time_s: 0.05
      sample_policy: interpolate_at_frame_center

    subblocks:
      duration_s: 1.0
      id_format: subblock_{index:06d}

    mapping:
      source.x_position_as: x_pointing_as
      source.y_position_as: y_pointing_as
      source.position_angle_deg:
        mode: constant
        value: 0.0

    outputs:
      outdir: Results/trajectory_campaigns/provider_sample_first60s
      write_trace_prescriptions: true
      write_render_prescriptions: true
      write_inference_prescriptions: false
```

This schema intentionally separates trajectory ingestion, window selection,
frame sampling, sub-block partitioning, and mapping into renderer-facing
parameter keys.

## Generated artifacts

The first implementation should write:

- campaign `manifest.json`
- copied or normalized campaign prescription
- normalized trajectory CSV for the selected input trajectory
- one sub-block folder per generated sub-block
- one explicit frame-truth CSV per sub-block
- optional renderer prescription for each sub-block
- optional inference prescription stub for each sub-block

Each sub-block trace CSV should follow the existing explicit trace contract:

```text
frame_index,time_s,source.x_position_as,source.y_position_as,source.position_angle_deg
```

The `time_s` column may be either local to the sub-block or absolute within the
observation. The first implementation should choose one convention and record it
clearly in the manifest. A useful convention is:

- sub-block trace CSV `time_s`: local time from sub-block start
- campaign manifest: records `observation_start_time_s` and each sub-block's
  `start_time_s`, `end_time_s`

This keeps existing single-sub-block plots readable while preserving global
observation timing in the manifest.

## Campaign manifest sketch

A campaign manifest should include enough information to reproduce and audit the
campaign.

Example structure:

```json
{
  "schema_version": "trajectory_observation_campaign_manifest.v1",
  "created_at": "...",
  "generator_id": "examples/recipes/trajectory_observation_campaign.py",
  "inputs": {
    "config_path": "...",
    "trajectory_path": "...",
    "trajectory_metadata_path": "..."
  },
  "trajectory": {
    "normalized_path": "trajectory/normalized_trajectory.csv",
    "columns": {
      "time_s": "time_s",
      "x": "x_pointing_as",
      "y": "y_pointing_as"
    },
    "units": {
      "x": "arcsec",
      "y": "arcsec"
    },
    "coordinate_convention": "TBD",
    "hash": "..."
  },
  "window": {
    "start_time_s": 0.0,
    "duration_s": 60.0,
    "end_time_s": 60.0
  },
  "frames": {
    "cadence_hz": 20.0,
    "exposure_time_s": 0.05,
    "n_frames_total": 1200,
    "sample_policy": "interpolate_at_frame_center"
  },
  "subblocks": {
    "duration_s": 1.0,
    "count": 60,
    "entries": [
      {
        "subblock_id": "subblock_000000",
        "index": 0,
        "start_time_s": 0.0,
        "end_time_s": 1.0,
        "n_frames": 20,
        "trace_csv": "subblocks/subblock_000000/trace/frame_truth.csv",
        "render_prescription": "subblocks/subblock_000000/render/prescription.yaml",
        "inference_prescription": null
      }
    ]
  }
}
```

## Jitter and smear modeling

The trajectory-driven campaign layer should distinguish between at least three
motion concepts.

### Frame-to-frame trajectory

This is the pointing value assigned to each frame, typically evaluated at frame
center. This is the first implementation target.

### High-frequency jitter

This is motion that is not resolved by the trajectory samples or is treated as a
statistical process. In early experiments, this can be represented as additional
additive trace effects or as a detector/optical blur approximation.

### Within-exposure smear

This is motion during the exposure integration time. It is not necessarily
captured by one frame-center pointing value.

Candidate treatments include:

- ignore smear in first implementation
- approximate smear as an `ApplyJitter` or convolution-like detector layer
- render multiple intra-exposure samples and average them
- use a line-smear kernel when the intra-frame motion is approximately linear

The first issue should not decide the final physical treatment. It should record
enough metadata that later smear/jitter experiments can be layered onto the same
campaign structure.

## Static model-error scenarios

A primary use of trajectory-driven campaigns is to test how static model errors
propagate through a sequence of sub-blocks.

Candidate static mismatch scenarios include:

- biased plate scale in the inference model
- biased optical aberration coefficients
- detector pixel-offset knowledge error
- detector response knowledge error
- flux or contrast mismatch
- unmodeled jitter/smear

These scenarios should eventually be represented as campaign variants:

```text
provider_first60s_nominal/
provider_first60s_plate_scale_bias_ppm10/
provider_first60s_pixel_offsets_mismatch_seed001/
provider_first60s_smear_unmodeled/
```

The first campaign generator does not need to implement all scenarios. It should
leave a clean place in the prescription and manifest to record scenario labels
and fixed truth/assumed-model differences.

## Relationship to estimability studies

Trajectory-driven campaigns are complementary to, but distinct from,
shared-parameter estimability studies.

Estimability studies ask questions such as:

- Which shared parameters are identifiable from a block or sequence of blocks?
- Which parameters are degenerate with frame-varying registration?
- Which nuisance parameters should be actively inferred, marginalized, or held
  fixed?
- How does the Fisher information or Schur complement change with trajectory,
  noise, or cadence?

Trajectory-driven campaigns ask questions such as:

- Can the implemented algorithm run on a realistic observation-like sequence?
- Do recovered sub-block products reduce sensibly across time?
- Does noise average down as expected?
- How do static model errors appear in proposal-facing diagnostics?
- What artifacts and summaries are needed for the observation-level estimator?

The two tracks should inform each other, but neither should block the first
implementation of the other.

## Proposal-facing demonstration path

A successful near-term demonstration could be:

1. ingest the provider trajectory
2. select the first 60 seconds
3. generate 60 one-second sub-block trace CSVs
4. render a smaller subset first, such as 5--10 sub-blocks
5. run registration-only inference on those rendered sub-blocks
6. aggregate recovered frame traces and sub-block summaries
7. plot recovered vs. true pointing over time
8. plot cumulative astrometric estimate versus number of sub-blocks
9. show approximate noise averaging with accumulated data
10. repeat under one or two static mismatch scenarios

This demonstration would support proposal language such as:

> We exercised the hierarchical inference workflow on a realistic 30-minute
> spacecraft pointing trajectory. Synthetic one-second observation sub-blocks
> were generated at the expected frame cadence, fit with the differentiable
> image model, and reduced into observation-level diagnostics. The study
> demonstrated stable recovery of frame-varying pointing terms and the expected
> improvement in astrometric precision as independent sub-blocks were
> accumulated, while also quantifying sensitivity to controlled static model
> errors.

The exact wording should wait until results are available, but this is the
shape of the claim the infrastructure is intended to support.

## Recommended first GitHub issue

Title:

```text
Add trajectory-driven observation campaign generation for sub-block inference demos
```

Suggested labels:

```text
design
pipeline
diagnostics
ready-for-agent
```

Suggested scope:

- add a design note under `docs/dev/trajectory_driven_observation_campaign_design.md`
- add an example trajectory data location and README/metadata template
- add a campaign-generation recipe or script that:
  - loads a provider trajectory CSV
  - normalizes required columns
  - selects a configurable time window
  - samples at a configurable frame cadence
  - partitions frames into one-second sub-blocks
  - writes explicit per-subblock frame-truth CSVs
  - writes a campaign manifest
  - optionally writes render prescription stubs for each sub-block
- add a small template prescription under `examples/recipes/trajectory_observation_campaign_template/`
- add tests for:
  - trajectory normalization
  - frame sampling/interpolation
  - sub-block partitioning
  - manifest path/provenance fields
  - generated trace CSV compatibility with existing trace loader expectations
- update docs or README references so the workflow is discoverable

Suggested non-goals:

- no full 30-minute default render
- no observation-level estimator implementation
- no full shared-parameter inference implementation
- no final jitter/smear physical model
- no mission-scale planet/no-planet simulation

## Open questions

Before proposal-grade use, resolve:

1. What are the exact trajectory units and coordinate conventions?
2. Are X/Y values directly mappable to `source.x_position_as` and
   `source.y_position_as`, or is a coordinate transform required?
3. Does the trajectory include high-frequency jitter, lower-frequency pointing,
   commanded attitude, measured attitude, or something else?
4. Should sub-block trace `time_s` be local to each sub-block or global to the
   observation?
5. What should be the first static mismatch scenario for proposal plots?
6. What sub-block summary schema should feed the observation-level estimator?
7. Which aggregation metrics should be treated as proposal-facing success
   criteria?

## Suggested phased implementation

### Phase 0: Data triage and metadata

- inspect the provider trajectory file
- confirm columns, units, sign conventions, and time base
- decide whether the file can be committed to the repo
- write a metadata sidecar and README

### Phase 1: Campaign trace generation

- implement trajectory ingestion and normalization
- implement window selection and frame sampling
- partition frames into sub-blocks
- write frame-truth CSVs and campaign manifest
- add tests around sampling and partitioning

### Phase 2: Render orchestration glue

- generate renderer prescriptions for selected sub-blocks
- support dry-run previews of expected outputs
- optionally add a small helper to print commands for rendering selected blocks
- keep actual rendering opt-in to avoid accidental large runs

### Phase 3: Inference orchestration glue

- generate inference prescriptions for rendered sub-blocks
- support a selected subset of sub-blocks for early experiments
- collect paths to recovered traces and truth comparison files

### Phase 4: Observation-level aggregation diagnostics

- aggregate recovered sub-block products
- compute cumulative mean/uncertainty diagnostics
- compute per-subblock residual and chi-squared summaries
- plot recovered pointing, residuals, and cumulative astrometric estimates

### Phase 5: Model-error and jitter/smear studies

- define campaign variants for controlled mismatch scenarios
- add jitter/smear options once the physical approximation is chosen
- compare nominal and mismatched campaigns using the same aggregation layer

## Summary

Trajectory-driven observation campaigns are the next practical bridge between
isolated sub-block tests and a proposal-relevant algorithm demonstration. The
first implementation should stay focused: use the provider trajectory to produce
a reproducible, well-documented sequence of explicit per-subblock traces and
manifests. Rendering, inference, aggregation, shared-parameter inference, and
mission-scale planet-detection studies can then build on that campaign scaffold
incrementally.

## Trajectory-Derived Smear Sidecars

Trajectory campaigns can optionally derive within-exposure smear from the same resolved trajectory used for frame-center truth. This is controlled by `subblocks.trajectory_processing.smear` in wrapper recipes, or by a trajectory-processing config passed to `run_trajectory_subblock_campaign.py`.

The existing trajectory files keep their original meanings:

- `frame_truth.csv` is frame-center truth for the renderer.
- `starting_guess_prediction.csv` is optimizer initialization only.
- `frame_smear_truth.csv` and `frame_smear_model.csv` are explicit within-exposure smear sidecars.
- `smear_provenance.json` records source trajectory hash, exposure convention, interpolation policy, plate scale, mismatch policy, and render policy.

When smear is enabled, the default model/inference smear is `matched`, so the model sidecar has minimal knowledge error relative to truth. Mismatch is opt-in through `inference.mode`, currently including `matched`, `scaled`, `angle_offset`, `constant`, and `disabled`.

Render handling has two deliberately scoped modes:

- `metadata_only` writes sidecars and plan summary fields without modifying detector layers.
- `subblock_constant_layer` computes one representative per-subblock line kernel and injects an existing `ApplyConvolution` detector layer in standalone trajectory render configs.

Per-frame dynamic convolution kernels, dynamic crop / ROI-origin realism, high-order WFE coupling, spectral/WFE/trajectory combined smokes, and production trajectory campaigns are deferred.

## Model-Split Provenance

Observation-bias and full-fidelity smoke plans now reference the shared `campaign_model_split.v1` contract. Trajectory mode still preserves the existing file semantics: `frame_truth.csv` is frame-center render truth, `starting_guess_prediction.csv` is optimizer initialization only, and smear truth/model sidecars remain separate artifacts.

When trajectory smear is enabled in the full-fidelity smoke, the model split records `trajectory_smear.enabled=true` and `mode=metadata_only`. This is provenance only for the first smoke; dynamic crop/ROI handling and per-frame dynamic smear kernels remain deferred.
