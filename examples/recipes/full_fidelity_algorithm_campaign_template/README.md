# Full-Fidelity Campaign Template

## Overview

This directory documents three related but intentionally different full-fidelity campaign configuration files.

The review config is the default human-review prescription. It builds a physically meaningful resolved truth/reference split without hidden fast clamps, enables spectral response components, uses review-scale WFE maps, and documents structured noise intent while still avoiding a production-scale campaign.

The smoke config is a tiny integration wiring check. It proves that the wrapper can translate and wire source, spectral, WFE, trajectory, smear metadata, Schur summaries, prior draws, and iterative updates through the existing observation-bias campaign runner.

The larger v1 file is a future schema/design skeleton. It is not runner-ready and must not be treated as an executable campaign prescription.

## File Roles

| File | Role | Executable today? | Runner |
| --- | --- | --- | --- |
| `full_fidelity_binary_iterative_review.yaml` | Resolved-system review config with explicit grids and review-scale WFE | Yes | `examples/scripts/run_full_fidelity_binary_iterative_campaign.py` |
| `full_fidelity_binary_iterative_review.annotated.yaml` | Review config with contract notes | Not the default input | Comments are for reviewers |
| `full_fidelity_binary_iterative_review_config_reference.md` | Human-readable review field contract | Documentation | Generated from schema registry |
| `full_fidelity_binary_iterative_review_config_reference.json` | Machine-readable review field contract | Documentation/tooling | Generated from schema registry |
| `full_fidelity_binary_iterative_smoke.yaml` | Tiny executable smoke config | Yes | `examples/scripts/run_full_fidelity_binary_iterative_campaign.py` |
| `full_fidelity_binary_iterative_smoke.annotated.yaml` | Smoke config with contract notes | Not the default input | Comments are for reviewers |
| `full_fidelity_binary_iterative_smoke_config_reference.md` | Human-readable smoke field contract | Documentation | Generated from schema registry |
| `full_fidelity_binary_iterative_smoke_config_reference.json` | Machine-readable smoke field contract | Documentation/tooling | Generated from schema registry |
| `full_fidelity_algorithm_campaign_v1.yaml` | Future schema skeleton/design target | No | No current runner consumes it |

## Executable Config Tiers

Use `full_fidelity_binary_iterative_review.yaml` for resolved-system review and `full_fidelity_binary_iterative_smoke.yaml` for tiny wiring validation. Both are size/config variants of the same executable schema:

- `experiment.kind: full_fidelity_binary_iterative`
- `experiment.schema_version: full_fidelity_binary_iterative.v1`

The wrapper accepts:

- `experiment.kind: full_fidelity_binary_iterative`
- `experiment.kind: observation_bias_campaign` for already translated replay/debug configs

Deprecated aliases `full_fidelity_binary_iterative_review` and
`full_fidelity_binary_iterative_smoke` may be accepted temporarily, but the
wrapper normalizes them to `full_fidelity_binary_iterative` and records the
alias only as provenance.

The wrapper rejects:

- `experiment.kind: full_fidelity_algorithm_campaign`

The wrapper stays thin. It translates the executable full-fidelity schema into `observation_bias_campaign` and delegates to `examples/scripts/run_observation_bias_campaign.py`. The translated config carries `source_campaign_kind: full_fidelity_binary_iterative` and forwards the model-split blocks into the existing observation-bias campaign machinery.

## Future Schema Skeleton

`full_fidelity_algorithm_campaign_v1.yaml` is a schema skeleton and design target. It describes eventual production architecture: richer observation windows, detector pixel offsets, flat fields, nonlinearity, dark current, knockdowns, richer spectral/optical settings, and production outputs.

It is intentionally not consumed by a runner yet. Do not copy skeleton-only blocks into the executable smoke config unless wrapper support is implemented and tested. Examples of future/deferred blocks include `detector.pixel_offsets`, `detector.flat_field`, `detector.nonlinearity`, `detector.dark_current`, top-level `smear`, `knockdowns`, and production `outputs`.

## Runtime Cost Vs Physical Fidelity

Runtime cost is controlled by explicit settings, not by the word full-fidelity.

Primary runtime-cost controls:

- `spectral_model.truth.n_lambda` and `spectral_model.inference.n_lambda`
- `spectral_model.truth.components.*.enabled` response components
- `high_order_wfe.truth.npix`
- `subblocks.n_subblocks`
- `subblocks.n_frames`
- `subblocks.reference_n_iter`
- `iterative.windows_per_draw`
- `iterative.subblocks_per_window`
- `observation_theta` state dimension

Primary physical-fidelity controls:

- target-aware source SED mode and target
- truth/inference spectral wavelength ranges and response components
- high-order WFE truth amplitude/map size and inference knowledge error
- trajectory trace source and jitter
- smear metadata mode
- noise mode
- active observation-theta parameters

`spectral_model.fast` is documented as a smoke-only shortcut. The current review config and smoke config use explicit `n_lambda` values instead of hidden clamping. If `fast: true` is introduced in an ad hoc smoke config, audit reports the clamp semantics: truth `n_lambda <= 7` and inference `n_lambda <= 5`.

## Trajectory, Subblocks, And Iterative Windows

When `experiment.iterative.enabled: true`,
`iterative.windows_per_draw * iterative.subblocks_per_window` is the canonical
total number of subblocks generated per prior draw. `subblocks.n_subblocks` is
optional in this mode; when present it must match the iterative product.
`experiment.subblocks.trace_source.window.start_s` selects the start of the
continuous trajectory interval. `trace_source.window.n_subblocks` is optional;
when present it must match the resolved total subblock count. The executable
review and smoke configs omit the redundant window value.

`experiment.subblocks.n_frames` controls how many frame centers are sampled in
each subblock. Frame times are generated from the selected trajectory window as
clusters within each subblock: the first cluster begins at `window.start_s`, the
next cluster begins one `subblock_duration_s` later, and so on. With small
`n_frames` values and one-second subblock spacing, selected-frame plots appear
as separated clusters rather than a continuous line. This is expected; it does
not imply a discontinuity in the source trajectory.

`experiment.iterative.windows_per_draw` and
`experiment.iterative.subblocks_per_window` describe how generated subblocks are
grouped into iterative update windows. For iterative-disabled configs,
`subblocks.n_subblocks` is the canonical total subblock count, defaulting
clearly to 1 when omitted by the existing runner.

Trajectory filtering is applied to the continuous trace before selected frame
times are sampled when `trace_source.processing.filter.apply_stage:
before_window`. Review plots label components by filter kind:

- High-pass: filtered is the high-pass residual; removed is the low-frequency
  trend, `raw - filtered`.
- Low-pass: filtered is the low-pass trend; removed is the high-frequency
  residual, `raw - filtered`.
- Band-pass: filtered is the in-band component; removed is the out-of-band
  component, `raw - filtered`.

The notebook and config audit report the resolved subblock plan, filter
provenance, per-subblock frame timing, and per-subblock linear-fit residuals.

## Render Noise And Inference Variance

Render/data noise and the inference likelihood variance are separate controls.

`experiment.subblocks.noise` accepts legacy scalar values (`enabled`,
`disabled`, `inherit`) and the structured review mapping. The structured mapping
uses `enabled`, `shot_noise`, `read_noise`, `dark_current`,
`use_detector_read_noise`, `read_noise_electrons`,
`use_detector_dark_current`, `dark_current_e_per_s`, `variance_floor`,
`write_variance`, and `seed_policy`. Legacy values are normalized internally and
written to provenance.

Structured noise is active in the campaign path. The observation-bias planner
patches generated render templates under `experiment.noise` with term-specific
settings before dry-run/execution. `shot_noise` is also written as legacy
`photon_noise` for render-template compatibility. The subblock command uses
`--noise inherit` for structured requests so the term-specific template
settings are not overwritten by a coarse CLI flag.

Read-noise amplitude provenance is explicit. `read_noise_electrons` wins when
set; otherwise the audit resolves the value from the detector config/spec and
records the source. Dark-current provenance follows the same rule with
`dark_current_e_per_s`; if dark current is enabled, the expected variance uses
`dark_current_e_per_s * exposure_time_s`.

The resolved-system review notebook renders the main noise audit through the
resolved truth-system Binder path. It resolves `psf_npix` from any explicit
review override first, then from `truth_system.optics.psf_npix`, then from the
system preset/default. The recommended minimum render size for this audit is
160 pixels; the review config normally resolves to `psf_npix: 256`. Any display
crop is applied after the full image, noisy image, residual, and variance maps
are rendered, and notebook output records both `rendered_psf_npix` and
`displayed_crop_npix`.

Exposure time for the review render is resolved from
`experiment.subblocks.exposure_time_s` before source/system defaults. The same
resolved value is used for noise rendering and expected variance diagnostics.
Shot-noise variance follows the rendered model counts, read-noise variance is
fixed at `read_noise_electrons**2` per pixel, and dark-current variance scales
as `dark_current_e_per_s * exposure_time_s`. High image/colorbar count levels
should therefore be interpreted only after checking the printed exposure-time
provenance and image units.

Dry-run/execution writes noise provenance under the run root, including the
original request, normalized render terms, read/dark-current amplitude sources,
variance-floor source, and resolved `use_render_variance` behavior. The notebook
review and strict audit use the same normalized object as the campaign path.

`experiment.subblocks.noise.variance_floor` is the canonical variance floor. The
legacy `experiment.subblocks.variance_floor` field is deprecated and should be
removed from new configs. The value is a variance floor, not a read-noise sigma.

`write_variance` and `use_render_variance` control different stages:
`write_variance: true` asks the renderer to produce variance artifacts when
supported; `use_render_variance` controls the inference objective. `true` requests
`variance_model: provided_cube`; `false` uses the data/floor variance model;
`auto` resolves explicitly from the normalized render-noise request and is
reported in provenance/audit output.

## Data/Inference Split

The executable configs create deliberate truth/reference mismatches:

- truth spectral grid/range may differ from inference grid/range;
- review config has no hidden spectral fast clamp;
- truth high-order WFE maps are synthetic and nonzero;
- inference high-order WFE uses `knowledge_error` rather than exact truth;
- trajectory-derived `subblock_constant_layer` smear is rendered in truth/data
  templates and matched in inference templates.

The split contract is written under the run root during dry-run/execution:

- `model_split/model_split.json`
- `model_split/model_split_summary.json`
- `template_hashes.csv`
- `campaign_plan.json` fields `model_split` and `template_hashes`
- per-row hash/path columns in `subblock_plan.csv`, `expected_outputs.csv`, and `iterative_plan.csv`

Trace and render templates use the truth system config. Inference templates use the reference/inference system config. Spectral shape and high-order map pixels are not optimizer-visible parameters.

## Component Documentation

Use the companion files for field-by-field review:

- `full_fidelity_binary_iterative_review.annotated.yaml` explains the review tier contract inline.
- `full_fidelity_binary_iterative_review_config_reference.md/json` lists valid values, implemented status, consumers, runtime/fidelity/provenance effects, omit safety, and notes.
- `full_fidelity_binary_iterative_smoke.annotated.yaml` and `full_fidelity_binary_iterative_smoke_config_reference.md/json` document the tiny smoke tier.

Smoke-only shortcuts and labels:

- `n_draws` documents intent but is not consumed today; `prior_draws.n_cases` controls prior cases.
- `spectral_model.fast` is smoke_only and forbidden in the review config.
- disabled response components, small WFE maps, low frame counts, and low optimizer iterations keep smoke artifacts small.
- `forecast.enabled: false` and `eigenbasis.enabled: false` keep executable configs focused on current wiring.

Fields forwarded unchanged into observation-bias include:

- `spectral_model`
- `high_order_wfe`
- `subblocks`
- `iterative`
- `seeding`
- `observation_theta`
- `prior_draws`
- `truth_realization`
- `eigenbasis`
- `forecast`

Fields consumed directly by the wrapper include:

- `kind`
- `schema_version`
- `run_name`
- `seed`
- `source_kind`
- `target`
- `n_cases`
- `system_preset`
- `detector_overrides`

## Detector Layer Policy

Executable full-fidelity review and smoke configs now default to
`SHERA_FLIGHT_3P_CONV`. That preset supplies the detector realism layers used by
the full-fidelity path in a fixed order:

1. `pixel_mtf`
2. `diffusion`
3. `pixel_offsets`
4. `pixel_response`
5. `jitter`
6. `smear`

The full-fidelity wrapper preserves that preset when translating to the
observation-bias runner by writing both top-level `system.preset` and
`experiment.system_preset`; ordinary observation-bias campaigns keep the legacy
default unless they explicitly select a different preset.

Campaign scripts do not dynamically insert those standard layers when this
preset is selected. Instead, the resolved system config is patched before
system instantiation:

- `experiment.detector_overrides.layers.<name>.action: update` deep-merges
  fields into the named detector layer.
- `action: remove` removes the named layer while preserving unrelated layer
  order.
- `action: disable` is currently equivalent to removal.
- missing layers fail unless `allow_missing: true` is set.

The review and smoke configs keep the named `jitter` layer but reduce it to
`sigma_x = sigma_y = 0.001` detector pixels. This preserves a stable layer name
for audits/notebooks without adding a large extra pointing blur. Audits warn if
`jitter` exceeds `0.05` detector pixels while trajectory-derived frame truth or
smear is enabled, because that may double-count pointing blur.

Smear policy is explicit because `SHERA_FLIGHT_3P_CONV` contains a nonzero
default line-smear layer. `render.mode: disabled` removes the named `smear`
layer. `render.mode: metadata_only` may still write smear sidecars, but it also
removes the named `smear` layer so no rendered smear is applied; this is a
diagnostic/debug mode. `render.mode: subblock_constant_layer` fits X/Y pointing
over each subblock, scales the fitted slope to one rendered frame exposure,
converts that displacement to detector pixels, and patches the named `smear`
layer with trajectory-derived `length`, `theta_deg`, `sigma_perp`,
`kernel_size`, and `units: detector_pix`.

The same smear kernel is used for every rendered frame/image in a subblock.
Per-subblock render and inference templates are written under each trajectory
subblock artifact directory and their paths/hashes are recorded in plan CSVs.
`inference.mode: matched_subblock_constant` patches the inference template with
the same subblock-level kernel; `inference.mode: disabled` removes the smear
layer from the inference/reference template. `render.mode: per_frame` and
inference modes such as `solve_subblock_smear` remain future/deferred and fail
clearly if requested before implementation.

## Preset Migration Table

| Script/config | Old default preset | New default preset | Migration status | Reason |
| --- | --- | --- | --- | --- |
| `full_fidelity_binary_iterative_review.yaml` | `SHERA_FLIGHT_3P` | `SHERA_FLIGHT_3P_CONV` | Migrated | Full-fidelity review should exercise the detector-realism preset and explicit layer policy. |
| `full_fidelity_binary_iterative_smoke.yaml` | `SHERA_FLIGHT_3P` | `SHERA_FLIGHT_3P_CONV` | Migrated | Smoke validates the same detector preset family and rendered subblock smear path at tiny scale. |
| `run_full_fidelity_binary_iterative_campaign.py` | hard-coded `SHERA_FLIGHT_3P` fallback | `DEFAULT_FULL_FIDELITY_SYSTEM_PRESET` (`SHERA_FLIGHT_3P_CONV`) | Migrated | Wrapper fallback now matches executable full-fidelity configs. |
| `audit_full_fidelity_config.py` | documented `SHERA_FLIGHT_3P` fallback | `DEFAULT_FULL_FIDELITY_SYSTEM_PRESET` (`SHERA_FLIGHT_3P_CONV`) | Migrated | Audit reports base, overridden, and smear-policy detector stacks. |
| `full_fidelity_resolved_system_review.ipynb` / notebook backend | inherited wrapper fallback | `DEFAULT_FULL_FIDELITY_SYSTEM_PRESET` (`SHERA_FLIGHT_3P_CONV`) | Migrated via backend | Notebook review uses the same resolver/override path. |
| `run_observation_bias_campaign.py` | `SHERA_FLIGHT_3P` | `SHERA_FLIGHT_3P` | Legacy baseline preserved | General observation-bias campaigns are not automatically full-fidelity migrations. |
| `run_trajectory_subblock_campaign.py` | `SHERA_FLIGHT_3P` | `SHERA_FLIGHT_3P` | Legacy baseline preserved | Standalone trajectory campaigns retain historical defaults unless a config/CLI selects CONV. |
| `run_single_star_calibration_demo.py` | `SHERA_FLIGHT_3P` | `SHERA_FLIGHT_3P` | Legacy baseline preserved | Single-star calibration demo is outside the new full-fidelity path. |
| `run_single_star_both_wfe_campaigns.py` | `SHERA_FLIGHT_3P` | `SHERA_FLIGHT_3P` | Legacy baseline preserved | Historical WFE comparison baseline is preserved. |
| `run_obs_subblock_study.py` | template-driven | template-driven | Not migrated | Subblock runner consumes already prepared templates. |

## Spectral And WFE Policy Notes

The default spectral photometry mode is `preserve_detected_flux_parameters`.
With this policy, `source.log_flux_total` and `source.contrast` are treated as
detected, post-response, band-integrated scalar parameters. SED, detector QE,
and M2/filter response change normalized spectral weights and record flux
factors in provenance; they do not silently rescale the scalar flux parameters.
Throughput-aware scalar photometry modes are documented as future/deferred in
the field registry until implemented and tested.

High-order WFE knowledge-error maps are additive correlated high-order OPD
errors. By default, configured low-order Zernike modes are removed from the
additive error before it is added to the truth high-order map. This avoids
double-counting low-order WFE uncertainty, which is already represented by the
low-order Zernike coefficient state.

The WFE deck separates low-order state from high-order map realism. For each
mirror, a raw truth OPD map is generated and piston/tip/tilt are removed. The
configured low-order modes, normally Noll Z4-Z11, are fit from that raw map and
stored as low-order coefficient arrays where array index 0 maps to Z4. Their
Zernike reconstruction is subtracted from the raw map to form the stored
high-order truth residual OPD. The reference/inference high-order map is then
the high-order truth residual plus a separate high-order knowledge-error
residual. That error residual also has piston/tip/tilt and configured low-order
Zernike modes removed by default.

Review plots should therefore show stored low-order coefficients separately
from residual low-order projections. Near-zero residual projection bars are
expected: they are leakage/orthogonality diagnostics showing that low-order
modes were removed from the high-order residual maps. Meaningful low-order WFE
bias should be read from the stored truth/reference/error coefficient table,
not from a refit to already-filtered high-order residual OPD maps.

The system preset defines the base source, optics, detector, wavelength
defaults, low-order WFE coefficient arrays, and detector layer stack. The
spectral deck patches source wavelength fields after system resolution.
High-order WFE patches truth/reference optics configs after base resolution.
Observation theta selects active slow parameters; it does not redefine the
complete physical basis. `subblocks.exposure_time_s` may override source
exposure time for campaign rendering.

## Running Audit And Smoke

Review-config strict audit:

```bash
PYTHONPATH=src python3 examples/scripts/audit_full_fidelity_config.py \
  --config examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_binary_iterative_review.yaml \
  --outdir Results/full_fidelity_config_audit/review_v1 \
  --strict
```

Smoke static audit:

```bash
PYTHONPATH=src python3 examples/scripts/audit_full_fidelity_config.py \
  --config examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_binary_iterative_smoke.yaml \
  --outdir Results/full_fidelity_config_audit/smoke_v0
```

Dry-run first:

```bash
PYTHONPATH=src python3 examples/scripts/run_full_fidelity_binary_iterative_campaign.py \
  --config examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_binary_iterative_smoke.yaml \
  --run-name full_fidelity_binary_iterative_smoke_dryrun \
  --dry-run \
  --no-resource-time
```

Optional tiny execution:

```bash
PYTHONPATH=src python3 examples/scripts/run_full_fidelity_binary_iterative_campaign.py \
  --config examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_binary_iterative_smoke.yaml \
  --run-name full_fidelity_binary_iterative_smoke_exec \
  --max-workers 1 \
  --no-resource-time
```

Aggregate-only replay after a completed execution:

```bash
PYTHONPATH=src python3 examples/scripts/run_full_fidelity_binary_iterative_campaign.py \
  --config examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_binary_iterative_smoke.yaml \
  --run-name full_fidelity_binary_iterative_smoke_exec \
  --aggregate-only \
  --no-resource-time
```

Validation harness:

```bash
PYTHONPATH=src python3 examples/scripts/validate_full_fidelity_binary_iterative_smoke.py \
  --config examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_binary_iterative_smoke.yaml \
  --results-root Results/full_fidelity_validation \
  --stage dry-run
```

## Interpreting Generated Artifacts

After audit, inspect:

- `config_audit.md` for the concise human review report;
- `config_audit.json` for machine-readable classifications;
- `translated_observation_bias_config.json` to see exactly what the wrapper delegates;
- `field_reference.csv` or `field_reference.json` for field-level review;
- `resolved_component_summary.json` for effective spectral clamp and mismatch summaries.

After dry-run, inspect the run root:

- `resolved_config.json` to confirm the translated observation-bias config;
- `campaign_plan.json` for case/subblock/iterative wiring;
- `model_split_summary.json` and `model_split/model_split.json` for truth/reference hashes and component summaries;
- `templates/render_template.json` and `templates/inference_template.json` to confirm truth vs reference system configs;
- `subblock_plan.csv`, `expected_outputs.csv`, and `iterative_plan.csv` for per-row plan wiring;
- `template_hashes.csv` for reproducible template identity.

## Resolved-system Review Notebook

Use `examples/notebooks/full_fidelity_resolved_system_review.ipynb` when you want an interactive, cell-by-cell review of the executable review config before running a larger campaign.

Run it from the repository root:

```bash
PYTHONPATH=src jupyter lab examples/notebooks/full_fidelity_resolved_system_review.ipynb
```

or:

```bash
PYTHONPATH=src jupyter notebook examples/notebooks/full_fidelity_resolved_system_review.ipynb
```

The notebook checks the translated observation-bias config, resolved base/truth/inference systems, source wavelength grids and weights, flux-parameter preservation, high-order WFE decomposition, stored low-order Zernike coefficients, high-order residual projection leakage, optics preset fields, detector layer/calibration-map presence, campaign noise-path audit, the configured Airbus trajectory segment, configured trajectory filtering, trace-jitter wiring, and a compact reviewer dashboard.

Optional notebook artifacts are written under:

```text
Results/full_fidelity_resolved_system_review/<run_label>/
```

Inspect `resolved_*_system.yaml`, `model_split_summary.json`, `spectral_review_tables.csv`, `*_review_summary.json`, and `config_review_notes.md` if `WRITE_ARTIFACTS=True`.

The notebook does not launch a production campaign, run optimization, silently change the review/smoke configs, or implement dynamic smear physics. Trajectory filtering is used only when `experiment.subblocks.trace_source.processing.filter.enabled: true` or the legacy `experiment.subblocks.trajectory_processing.filter.enabled: true` path is set.

## Trajectory Filtering

Trajectory mode supports `none`, `low_pass`, `high_pass`, and `band_pass` filters in `experiment.subblocks.trace_source.processing.filter`. The canonical source schema is `source.kind: csv` with `source.format: airbus_xyz_arcsec`; legacy `source.kind: airbus_csv` remains accepted.

The implemented method is `bessel` via SciPy second-order sections. A Bessel filter is used when preserving the time-domain trajectory shape is more important than achieving the steepest possible cutoff. For offline preprocessing, `zero_phase: true` uses forward/backward filtering to avoid phase lag; `zero_phase: false` uses causal filtering and records expected phase/group delay in provenance.

Cutoff periods are frequency conveniences, not time constants: `cutoff_hz = 1 / cutoff_period_s`. For a high-pass filter with `cutoff_period_s: 15.0`, slower trends are suppressed and faster residual motion is preserved. For a low-pass filter the same period preserves slower trends and suppresses faster residuals. Band-pass filters require `low_cutoff_hz < high_cutoff_hz`; with periods, `low_cutoff_period_s` is the longer-period lower-frequency edge and `high_cutoff_period_s` is the shorter-period higher-frequency edge.

Filtering defaults to `apply_stage: before_window`, which loads and filters the full mapped trajectory before selecting frame windows. This avoids short-window edge artifacts from zero-phase padding. Enabled filtering writes `trajectory_raw.csv`, `trajectory_filtered.csv`, `trajectory_filter_provenance.json`, `trajectory_filter_summary.csv`, and a diagnostic plot beside the trajectory artifacts. Per-subblock `frame_truth.csv` is always the filtered truth when filtering is enabled; `frame_truth_unfiltered.csv` is written only when `write_unfiltered_comparison: true`.

## Intentionally Omitted From Smoke

The executable smoke may omit these future full-fidelity fields:

- detector pixel-offset maps;
- flat-field maps;
- nonlinearity;
- dark current;
- full 30-minute observation settings;
- per-frame dynamic smear kernels;
- dynamic crop / ROI-origin modeling;
- active high-order map inference;
- eigenbasis iterative update modes;
- production-scale forecast settings;
- proposal-grade plots.

The smoke should focus on wiring source target, spectral truth/reference split, high-order WFE truth/reference split, trajectory trace, optional smear metadata, Schur summary, tiny iterative update, provenance, and aggregate-only replay.

## Deferred Work

Deferred implementation areas remain explicit non-goals for this directory:

- make `full_fidelity_algorithm_campaign_v1.yaml` executable;
- implement detector pixel-offset or flat-field decks;
- implement dynamic crop / ROI-origin realism;
- implement per-frame dynamic smear kernels;
- implement active high-order WFE map inference;
- implement a full Bayesian recursive filter;
- tune science results or launch production campaigns.

## Common Pitfalls

Do not pass `full_fidelity_algorithm_campaign_v1.yaml` to the smoke wrapper. The wrapper will reject it because it is a design skeleton.

Do not assume `spectral_model.fast` is a vague label. It is implemented and clamps effective spectral grids. Set it deliberately and still review explicit `n_lambda`, wavelength range, and response components.

Do not add skeleton-only fields to the smoke config expecting them to work. The wrapper warns for future-only copied blocks, detector pixel-offset/flat-field blocks, unsupported dynamic smear modes, and high-order map-pixel observation-theta requests.

Do not interpret the tiny smoke output as proposal-grade performance. The configuration is intentionally small so reviewers can validate wiring and artifacts quickly.
