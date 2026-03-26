# Observation Sub-Block Simulation Contract

Status: Phase 4 simulation contract (renderer + separate trace-builder).

## 1. Purpose and scope

Observation sub-block simulation provides a stable explicit-trace boundary for
short time-series image generation:

1. generate per-frame trace rows
2. render one central-field image per frame from those rows

This layer bridges single-frame simulation and future multi-frame inference,
without introducing a full time-series framework.

In scope:

- explicit per-frame trace CSV contract
- configurable supported per-frame varying keys
- additive trace effects around an anchor value
- per-frame rendering into one `(n_frame, ny, nx)` FITS cube
- manifest + per-frame truth CSV outputs

Out of scope:

- multi-frame inference changes
- multi-ROI mission payloads
- structural per-frame system rebuilds
- arbitrary expression language for motion/effects

## 2. Architectural boundary

The architecture remains intentionally split:

- trace construction: `examples/recipes/observation_subblock_trace.py` +
  `src/dluxshera/utils/obs_subblock_trace_builders.py`
- rendering: `examples/recipes/observation_subblock.py`

Renderer input stays explicit: canonical CSV trace rows.

## 3. Renderer contract

### 3.1 Inputs

- resolved `system` config
- `experiment.kind: observation_subblock`
- shared truth overrides (`experiment.truth`)
- `experiment.observation_subblock.varying_keys` (optional; defaults to v1 trio)
- trace input:
  - `experiment.observation_subblock.trace.format` (currently `csv`)
  - `experiment.observation_subblock.trace.path`
- validation toggles:
  - `validate.require_contiguous_frame_index` (default `true`)
  - `validate.require_monotonic_time` (default `true`)

### 3.2 Outputs

- timestamped FITS cube
- timestamped frame-truth CSV
- `manifest.json`

### 3.3 Core frame execution model

Per frame:

1. read configured varying-key values from trace row
2. split overrides into primitive and derived key sets
3. apply primitive overrides
4. refresh derived values
5. re-apply explicit derived overrides (explicit derived precedence)
6. render frame through Binder
7. append rendered image + resolved truth row

This preserves explicit derived frame updates after refresh.

## 4. Supported varying-key policy

Observation-subblock simulation supports a defined non-structural key family.

### 4.1 Allowed scalar keys

- `source.x_position_as`
- `source.y_position_as`
- `source.position_angle_deg`
- `source.separation_as`
- `source.contrast`
- `source.log_flux_total`
- `optics.plate_scale_as_per_pix`

### 4.2 Allowed indexed vector components

- `optics.primary.zernike_coeffs_nm[i]`
- `optics.secondary.zernike_coeffs_nm[i]`

### 4.3 Rejected keys

- unsupported keys outside the allowed family
- structural keys (when resolved spec is available and marks structural)
- malformed indexed syntax
- out-of-bounds indexed components (when store shape is available)

## 5. Key-address syntax

One syntax is used across config, trace-plan keys, CSV columns, manifests, and
validation:

- scalar: `a.b.c`
- indexed component: `a.b.c[index]`

Examples:

- `source.x_position_as`
- `optics.primary.zernike_coeffs_nm[3]`

## 6. Trace schema and validation

### 6.1 Required columns

- `frame_index`
- `time_s`
- one column per applied `varying_keys`

### 6.2 Hard errors

- missing trace path / invalid path
- invalid `experiment.kind`
- missing required trace columns
- duplicate `frame_index`
- non-finite values in required numeric columns

### 6.3 Configurable checks

- contiguous `frame_index` (`require_contiguous_frame_index`)
- monotonic non-decreasing `time_s` (`require_monotonic_time`)

Trace rows are sorted by `frame_index` before downstream use.

### 6.4 Extra columns

Extra CSV columns are preserved in truth outputs and metadata. They do not
drive rendering unless listed in `varying_keys`.

## 7. Trace-builder contract

Set `experiment.kind: observation_subblock_trace` and configure:

- `n_frames`
- `dt_s`
- `varying_keys`
- `trace_plan` mapping per varying key
- optional seed (`experiment.seed` or trace block seed)

Trace-builder output remains canonical explicit CSV for renderer consumption.

## 8. Anchor/base semantics

For each varying key:

1. if `base` is provided, anchor = `base`
2. else anchor comes from resolved, refreshed nominal store value

Final generated value per frame:

`anchor + sum(additive_effects)`

If any key omits `base`, the trace recipe requires enough config to resolve a
system/store anchor. If all keys provide `base`, experiment-only trace configs
work without a `system` block.

## 9. Additive effects model

Each key can specify zero or more additive effects:

- `constant_offset` (`offset`)
- `linear_drift` (`start`, `rate_per_s`)
- `random_walk` (`start`, `sigma_step`)
- `iid_jitter` (`center`, `sigma`)
- `explicit` (`values`) for direct per-frame additive series

Effects are summed with the anchor. Randomness uses deterministic child seeds
derived from global seed + key + effect index + effect kind.

## 10. Manifest metadata

Renderer manifests include:

- `varying_keys`
- `applied_varying_keys`
- optional `requested_varying_keys`
- trace metadata and artifact relative paths

Trace-builder manifests include:

- `varying_keys`
- `applied_varying_keys`
- resolved per-key anchors
- normalized trace spec/effects

## 11. Output artifact layout

Renderer outputs:

`<outdir>/<run_name_or_timestamp>/`

- `<file_prefix>_<timestamp>_cube.fits`
- `<file_prefix>_<timestamp>_frame_truth.csv`
- `manifest.json`

Trace-builder outputs:

`<outdir>/<run_name_or_timestamp>/`

- `<file_prefix>_<timestamp>_frame_truth.csv`
- `manifest.json` (optional via config)

## 12. Backward compatibility

v1-style `x/y/PA` workflows still run:

- renderer defaults to v1 keys when `varying_keys` is omitted
- trace loader defaults to v1 required varying keys when none are supplied
- legacy trace-builder `keys: {<key>: {mode: ...}}` schema is still accepted
  (preferred schema is `trace_plan`)

## 13. Open follow-ons (Phase 5+)

- formal key registry ownership and extension policy
- richer trace formats (currently CSV only)
- multi-ROI outputs
- simulation/inference mismatch handling at block level
- tighter docs/tutorials around generalized key workflows
