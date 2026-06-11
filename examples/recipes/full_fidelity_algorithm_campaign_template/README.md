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

Use `full_fidelity_binary_iterative_review.yaml` for resolved-system review and `full_fidelity_binary_iterative_smoke.yaml` for tiny wiring validation.

The wrapper accepts:

- `experiment.kind: full_fidelity_binary_iterative_review`
- `experiment.kind: full_fidelity_binary_iterative_smoke`
- `experiment.kind: observation_bias_campaign` for already translated replay/debug configs

The wrapper rejects:

- `experiment.kind: full_fidelity_algorithm_campaign`

The wrapper stays thin. It translates the executable full-fidelity review/smoke schemas into `observation_bias_campaign` and delegates to `examples/scripts/run_observation_bias_campaign.py`. The translated config carries `source_campaign_kind` matching the source tier and forwards the model-split blocks into the existing observation-bias campaign machinery.

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

## Data/Inference Split

The executable configs create deliberate truth/reference mismatches:

- truth spectral grid/range may differ from inference grid/range;
- review config has no hidden spectral fast clamp;
- truth high-order WFE maps are synthetic and nonzero;
- inference high-order WFE uses `knowledge_error` rather than exact truth;
- trajectory trace metadata is present while dynamic per-frame smear rendering is deferred.

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

The notebook checks the translated observation-bias config, resolved base/truth/inference systems, source wavelength grids and weights, flux-parameter preservation, high-order WFE maps and Zernike projections, optics preset fields, detector layer/calibration-map presence, disabled-vs-demo noise behavior, the configured Airbus trajectory segment, a diagnostic 15 s high-pass residual, trace-jitter wiring, and a compact reviewer dashboard.

Optional notebook artifacts are written under:

```text
Results/full_fidelity_resolved_system_review/<run_label>/
```

Inspect `resolved_*_system.yaml`, `model_split_summary.json`, `spectral_review_tables.csv`, `*_review_summary.json`, and `config_review_notes.md` if `WRITE_ARTIFACTS=True`.

The notebook does not launch a production campaign, run optimization, silently change the review/smoke configs, implement dynamic smear physics, or make high-pass filtering part of the production trajectory source. The high-pass section is diagnostic unless an explicit supported config path is added later.

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
