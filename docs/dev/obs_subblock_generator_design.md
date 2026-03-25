# Observation Sub-Block Generator Design (Phase 1 Contract)

Status: Phase 1 contract document, with Phase 3 notes on implemented workflow separation.

## 1. Purpose and Scope

The observation sub-block generator produces a short, time-ordered image stack from one resolved dLuxShera system configuration plus an explicit per-frame trace table.

It is the first bridge from current single-frame recipes to later multi-frame inference workflows.

Phase 3 workflow note:
- Trace construction is now implemented as a separate helper/recipe path
  (`examples/recipes/observation_subblock_trace.py` +
  `src/dluxshera/utils/obs_subblock_trace_builders.py`) that emits canonical
  explicit CSV traces for the renderer.
- Rendering remains in `examples/recipes/observation_subblock.py` and still
  consumes explicit CSV traces only.

In scope for v1:
- Generate one analysis-oriented, central-field image cube (single ROI stream).
- Accept explicit frame-by-frame traces as the primary control input.
- Support frame-varying `source.x_position_as`, `source.y_position_as`, and `source.position_angle_deg`.
- Keep all other parameters shared across the sub-block.
- Write a timestamped FITS cube, `manifest.json`, and per-frame truth table.

Out of scope for v1:
- Mission/downlink product formats and 5-ROI frame payloads.
- Built-in motion-model abstraction inside the renderer.
- Multi-frame inference or hierarchical posterior updates.
- Slow-drift optics/calibration updates (for example plate scale or Zernike drifts).

## 2. Relationship to the Larger Time-Domain Concept

In the long-term mission story, frame data (for example 20 Hz / 50 ms cadence) are grouped into short sub-blocks, then aggregated into larger products (observation, orbit, day, mission). This generator defines only the first grouping layer: frame -> sub-block.

The scope is intentionally narrow so later inference/aggregation layers can rely on a stable, explicit sub-block contract without locking in higher-level mission assumptions yet.

## 3. Core Contract

### 3.1 What the Generator Consumes

- Resolved `system` config (preset + overrides via existing config resolver path).
- `experiment` block for sub-block generation settings, truth overrides, noise settings, and output settings.
- Explicit frame trace table (file-backed or inline rows).
- Optional run metadata (notes, run label).

### 3.2 What the Generator Writes

- One timestamped FITS cube for rendered frames.
- One `manifest.json` describing config lineage, trace summary, and emitted artifacts.
- One per-frame truth table (`csv` in v1; `jsonl` reserved as future extension).

### 3.3 Shared Base State vs Frame Overrides

- Build a shared base truth store once:
  - `compose_forward_spec(system_cfg)`
  - `ParameterStore.from_spec_defaults(...)`
  - apply `experiment.truth` overrides
  - `refresh_derived(...)`
- For each frame, apply only v1 frame overrides from trace:
  - `source.x_position_as`
  - `source.y_position_as`
  - `source.position_angle_deg`
- All non-varying keys stay at shared base values for the whole sub-block.
- `experiment.observation_subblock.varying_keys` is advisory in v1:
  - if omitted, renderer defaults to applied `x/y/PA` keys
  - if provided, renderer may preserve it as requested metadata, but still applies fixed `x/y/PA` keys

## 4. Recommended Config Shape

Use canonical nested config style with top-level `system` + `experiment`.

```yaml
system:
  preset: SHERA_TESTBED_3P
  # optional system overrides (source/optics/detector) follow normal resolver rules

experiment:
  kind: observation_subblock
  seed: 42
  notes: "optional experiment-level note"

  truth:
    source:
      separation_as: 10.0
      log_flux_total: 14.0
      contrast: 3.0
      exposure_time_s: 0.05

  observation_subblock:
    varying_keys:               # optional metadata in v1; renderer still applies fixed x/y/PA keys
      - source.x_position_as
      - source.y_position_as
      - source.position_angle_deg
    trace:
      format: csv                # currently supported v1 input format
      path: path/to/frame_truth.csv
      # inline_rows: [...]       # optional tiny-test fallback; mutually exclusive with path
    validate:
      require_contiguous_frame_index: true   # default true; honored if set false
      require_monotonic_time: true           # default true; honored if set false

  noise:
    enabled: false
    photon_noise: true
    read_noise: false
    dark_current: false

  outputs:
    outdir: Results/observation_subblock
    file_prefix: obs_subblock
    frame_truth_format: csv
```

Config placement decisions:
- Base system definition lives under `system`.
- Shared truth state lives under `experiment.truth`.
- Sub-block generation controls live under `experiment.observation_subblock`.
- Output policy lives under `experiment.outputs`.

## 5. Frame-Truth Table Schema

Required fields (v1):

| Field | Type | Notes |
| --- | --- | --- |
| `frame_index` | int | 0-based, unique, contiguous after sort |
| `time_s` | float | Frame timestamp relative to sub-block start |
| `source.x_position_as` | float | Per-frame source x offset (arcsec) |
| `source.y_position_as` | float | Per-frame source y offset (arcsec) |
| `source.position_angle_deg` | float | Per-frame source PA (deg) |

Recommended validation rules:
- duplicate `frame_index` values are always invalid.
- non-finite required numeric values are always invalid.
- `frame_index` contiguous check defaults to enabled, and is controlled by
  `validate.require_contiguous_frame_index`.
- monotonic `time_s` check defaults to enabled, and is controlled by
  `validate.require_monotonic_time`.
- Required fields must be present for every row.

Extensibility rule:
- Optional extra columns are allowed and should be preserved in output truth products and manifest metadata.
- v1 renderer applies only `x/y/PA` overrides even if extra columns are present.
- Future slow-drift support can activate selected extra columns (for example `optics.plate_scale_as_per_pix`) without changing the core table format.

## 6. Output Artifact Layout

Recommended experiment output layout:

```text
Results/observation_subblock/<timestamp_or_run_name>/
  manifest.json
  obs_subblock_<YYYYMMDD-HHMMSS>_cube.fits
  obs_subblock_<YYYYMMDD-HHMMSS>_frame_truth.csv
```

Artifact roles:
- FITS cube: primary image data product, shape `(n_frame, ny, nx)`.
- `manifest.json`: run metadata, config lineage, schema version, frame counts, artifact paths.
- per-frame truth table: explicit rendered frame parameters keyed by frame index/time.

Minimum `manifest.json` fields (v1 recommendation):

| Key | Notes |
| --- | --- |
| `schema_version` | Start with `"obs_subblock_manifest.v1"` |
| `created_at` | ISO timestamp |
| `generator` | Script/module id (for example recipe path) |
| `system` | Preset label and/or resolved config hash |
| `varying_keys` | Applied frame-varying key list (`x/y/PA` in v1; compatibility alias) |
| `applied_varying_keys` | Explicit applied renderer key list |
| `requested_varying_keys` | Optional user-provided varying-key metadata (if provided in config) |
| `frame_count` | Number of rendered frames |
| `time_start_s` / `time_stop_s` | From trace table |
| `trace` | Trace source metadata (`format`, `path` or `inline`, extra columns) |
| `artifacts` | Relative paths for cube + truth table + manifest |

Timestamp naming:
- Use `%Y%m%d-%H%M%S` pattern to match existing recipe conventions.

Quick-look previews:
- Optional future-friendly artifact only (for example PNG montage). Not contract-required for v1.

## 7. Internal Execution Model (Conceptual)

```text
1) Resolve config -> system_cfg, experiment_cfg
2) Compose forward spec from system_cfg
3) Build shared base truth store from defaults + experiment.truth, then refresh deriveds
4) Load explicit frame trace (CSV) and validate required schema/rules
5) Create binder once from (system_cfg, forward_spec, base_store)
6) For each frame row:
   a) Build frame override dict from x/y/PA columns
   b) Apply overrides to base store and refresh derived values
   c) Render frame image via binder forward path
   d) Optionally apply configured observation noise
   e) Append image to cube and append resolved truth row
7) Write FITS cube, frame_truth.csv, and manifest.json
```

Important execution note:
- Keep trace parsing/validation independent from rendering logic so trace helpers can evolve separately.

## 8. Design Decisions Locked in This Phase

- Explicit per-frame traces are the primary v1 interface.
- Trace helper generators (for example linear drift/random walk/iid jitter) are separate utilities, not renderer internals.
- v1 output is one central-field image cube, not the eventual 5-ROI mission product.
- v1 applied frame-varying renderer parameters are exactly:
  - `source.x_position_as`
  - `source.y_position_as`
  - `source.position_angle_deg`
- Future slow drifts are anticipated by schema/config shape, but not activated in v1 rendering.

## 9. Open Questions and Future Extensions

- Slow drifts:
  - When to allow frame- or block-varying `optics.plate_scale_as_per_pix`.
  - Which low-order Zernike terms, if any, should be activated first.
- Multi-ROI payloads:
  - How to package 5-ROI cubes and ROI metadata while preserving a stable manifest contract.
- Noise policy:
  - Whether to keep v1 noise handling as recipe-level options only, or formalize noise provenance in manifest schema.
- Preview/dry-run:
  - Whether Phase 2 should include a no-render validation mode for trace/config checking.
- Inference integration:
  - How sub-block products map to later multi-frame inference inputs and priors handoff.
- Packaging path:
  - Whether the first implementation should live only in `examples/recipes/` or begin a new `src/dluxshera/apps/simulate/` namespace (roadmap points to apps, current repo has not created it yet).

## 10. Recommended Phase 2 Implementation Plan

### 10.1 Suggested Files

- `examples/recipes/observation_subblock.py` (primary CLI recipe entry point).
- `examples/recipes/obs_subblock_template/prescription.yaml` + `frame_truth.csv` (minimal template pair).
- `src/dluxshera/utils/obs_subblock_trace.py` (trace load/validate helpers).
- `src/dluxshera/utils/obs_subblock_io.py` (FITS/manifest/truth writers).

Notes:
- Keep first implementation close to current recipe patterns.
- Delay `src/dluxshera/apps/simulate/` introduction until a broader apps-layout decision is made.

### 10.2 Suggested Tests

- `tests/utils/test_obs_subblock_trace.py`:
  - required columns, index/time validation, extra-column pass-through.
- `tests/demos/test_observation_subblock_demo.py`:
  - smoke run on a tiny trace (`N=3`), verify artifacts exist and cube shape is correct.
- `tests/utils/test_obs_subblock_manifest.py`:
  - manifest contains expected fields and relative artifact paths.

### 10.3 Suggested Docs Follow-up in Phase 2

- Add recipe usage notes in docs/tutorials or examples README.
- Extend `docs/README.md` links if/when recipe and template are added.
- Add a short status note in `docs/dev/working_plan.md` when implementation lands.
