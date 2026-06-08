# Observation Bias Campaign Template

This recipe runs a small observation-level bias campaign, not a mission-scale
simulation. It derives the full primary and secondary Zernike coefficient
layout from the resolved system store by default, launches image-backed
`schur_summary` sub-block exports, then accumulates those summaries into one
physical-basis observation update per bias case.

Run a dry plan first:

```bash
PYTHONPATH=src python examples/scripts/run_observation_bias_campaign.py \
  --config examples/recipes/observation_bias_campaign_template/prescription.yaml \
  --results-root Results/observation_bias_campaign \
  --run-name full_zernike_bias_smoke_dryrun \
  --dry-run
```

Then run the smoke campaign:

```bash
PYTHONPATH=src python examples/scripts/run_observation_bias_campaign.py \
  --config examples/recipes/observation_bias_campaign_template/prescription.yaml \
  --results-root Results/observation_bias_campaign \
  --run-name full_zernike_bias_smoke
```

The observation state is stored in physical labels such as
`source.separation_as` and `optics.primary.zernike_coeffs_nm[0]`. Eigenmodes
are diagnostic transforms of the accumulated or posterior precision matrix;
they are not the native storage basis.

Seeding is explicit and deterministic. The template uses
`different_jitter_different_noise` with `base_seed: 42`, and each subblock plan
records `subblock_seed`, `trace_seed`, and `noise_seed`. Those seeds are passed
to the subblock runner as `--trace-seed` and `--render-seed`.

`prior_draws` is available for canonical-style prior-sampled reference states.
When enabled, the campaign expands wildcard sigma rules (for example
`optics.primary.zernike_coeffs_nm[*]`) onto resolved layout labels, draws
`theta_reference_offsets` from the configured normal priors, and writes
`prior_draws.csv` plus per-case prior draw metadata.

The `forecast` block extrapolates the measured image-backed summaries to a
30-minute observation target without running 1,800 Schur solves. `replicate`
tiles the actual summaries deterministically as an information accumulation
check. `fixed_information_score_noise` keeps template information fixed and
draws Fisher-style score noise for stochastic forecast rows.

The Zernike mask controls can narrow the system-derived coefficient list:
`include` selects a subset and `exclude` removes entries after selection. This
is useful for smoke tests without changing the layout machinery used by full
system runs.

Success is not recovering every M1/M2 coefficient independently. The useful
signal is whether the campaign identifies constrained and weak optical
combinations and whether the `source.separation_as` posterior moves as
expected.

Trajectory-backed smoke:

```bash
PYTHONPATH=src python examples/scripts/run_observation_bias_campaign.py \
  --config examples/recipes/observation_bias_campaign_template/trajectory_airbus_smoke.yaml \
  --results-root Results/observation_bias_campaign \
  --dry-run
```

`subblocks.trace_source.mode: trajectory` materializes shared Airbus-derived
`frame_truth.csv` and `starting_guess_prediction.csv` artifacts for each
subblock, then child commands pass those files to `run_obs_subblock_study.py`.
Binary trajectory mode defaults to X/Y/PA. IID jitter remains the default when
`trace_source` is omitted or set to `iid_jitter`.

## Binary Iterative Validation

`binary_iterative_smoke.yaml` is a parser/orchestration smoke. It uses IID jitter,
low-order WFE, physical-label summaries, and `iterative.update_mode:
physical_full` with forecast disabled.

`binary_iterative_cluster_validation.yaml` is the first bounded cluster
validation recipe. It asks whether two biased binary prior draws improve over
three repeated windows with two 20-frame subblocks per window. It keeps native
state in physical labels and writes per-window reference-update artifacts.

Dry-run before submitting:

```bash
PYTHONPATH=src python3 examples/scripts/run_observation_bias_campaign.py \
  --config examples/recipes/observation_bias_campaign_template/binary_iterative_cluster_validation.yaml \
  --results-root "$DLUX_RESULTS" \
  --run-name binary_iterative_cluster_validation_v1 \
  --dry-run
```

Execute bounded validation:

```bash
PYTHONPATH=src python3 examples/scripts/run_observation_bias_campaign.py \
  --config examples/recipes/observation_bias_campaign_template/binary_iterative_cluster_validation.yaml \
  --results-root "$DLUX_RESULTS" \
  --run-name binary_iterative_cluster_validation_v1 \
  --max-workers 1 \
  --resource-time auto
```

Aggregate after completion:

```bash
PYTHONPATH=src python3 examples/scripts/run_observation_bias_campaign.py \
  --config examples/recipes/observation_bias_campaign_template/binary_iterative_cluster_validation.yaml \
  --results-root "$DLUX_RESULTS" \
  --run-name binary_iterative_cluster_validation_v1 \
  --aggregate-only
```

Inspect `analysis/aggregate_status.json`, `analysis/output_inventory.csv`,
`analysis/missing_outputs.csv`, `analysis/iterative_window_diagnostics.csv`, and
per-window `cases/<case>/windows/window_XXX/iterative_reference_update.json`.
Diagnostics distinguish the full posterior update from the applied reference
update, which matters when `update_gain != 1`.

A checked-in Gattaca2 sbatch template is available at:

```text
examples/recipes/observation_bias_campaign_template/binary_iterative_cluster_validation.sbatch
```
