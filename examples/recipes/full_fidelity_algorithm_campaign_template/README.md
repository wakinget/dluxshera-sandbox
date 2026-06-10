# Full-Fidelity Campaign Template

## Overview

This directory documents two related but intentionally different full-fidelity campaign configuration files.

The executable file is a tiny integration smoke. It proves that the wrapper can wire a binary source, target-aware spectral truth/reference split, high-order WFE truth/reference split, trajectory trace source, smear metadata, Schur summaries, prior draws, and conservative iterative updates through the existing observation-bias campaign runner.

The larger v1 file is a future schema/design skeleton. It is not runner-ready and must not be treated as an executable campaign prescription.

The smoke config is a wiring/integration test, not a proposal-grade science campaign.

## File Roles

| File | Role | Executable today? | Runner |
| --- | --- | --- | --- |
| `full_fidelity_binary_iterative_smoke.yaml` | Current tiny executable smoke config | Yes | `examples/scripts/run_full_fidelity_binary_iterative_campaign.py` |
| `full_fidelity_binary_iterative_smoke.annotated.yaml` | Review-only annotated companion | Not the default input | Use only if manually selected; comments are for reviewers |
| `full_fidelity_binary_iterative_smoke_config_reference.md` | Human-readable smoke field reference | Documentation | Generated from audit/reference metadata |
| `full_fidelity_binary_iterative_smoke_config_reference.json` | Machine-readable smoke field reference | Documentation/tooling | Generated from audit/reference metadata |
| `full_fidelity_algorithm_campaign_v1.yaml` | Future schema skeleton/design target | No | No current runner consumes it |

## Executable Smoke Config

Use `full_fidelity_binary_iterative_smoke.yaml` for current validation.

The wrapper accepts:

- `experiment.kind: full_fidelity_binary_iterative_smoke`
- `experiment.kind: observation_bias_campaign` for already translated replay/debug configs

The wrapper rejects:

- `experiment.kind: full_fidelity_algorithm_campaign`

The wrapper stays thin. It translates the narrow smoke schema into `observation_bias_campaign` and delegates to `examples/scripts/run_observation_bias_campaign.py`. The translated config carries `source_campaign_kind: full_fidelity_binary_iterative_smoke` and forwards the model-split blocks into the existing observation-bias campaign machinery.

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

`spectral_model.fast` is consumed by `dluxshera.utils.campaign_model_split._build_spectral_deck`. When `true`, it clamps effective spectral grids to `truth.n_lambda <= 7` and `inference.n_lambda <= 5`. It is a smoke cost reducer and reduces spectral sampling fidelity. It is not a substitute for explicit `n_lambda`, wavelength range, and response-component settings.

## Data/Inference Split

The smoke config creates deliberate truth/reference mismatches:

- truth spectral grid/range differs from inference grid/range;
- `spectral_model.fast: true` further clamps effective spectral grids for smoke cost;
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

- `full_fidelity_binary_iterative_smoke.annotated.yaml` explains active blocks inline.
- `full_fidelity_binary_iterative_smoke_config_reference.md` lists field paths, consumers, runtime/fidelity/provenance effects, and omit safety.
- `full_fidelity_binary_iterative_smoke_config_reference.json` provides the same information for tooling.

Smoke-only shortcuts and labels:

- `n_draws` documents intent but is not consumed today; `prior_draws.n_cases` controls prior cases.
- `spectral_model.fast` is an implemented smoke cost reducer with explicit clamping semantics.
- disabled response components and `write_maps: false` keep artifacts and dependencies small.
- `forecast.enabled: false` and `eigenbasis.enabled: false` keep the smoke focused on wiring.

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

## Running The Smoke

Static audit only:

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
