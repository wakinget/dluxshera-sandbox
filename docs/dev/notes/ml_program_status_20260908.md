# ML Program Status Snapshot 2026-09-08

## Purpose

This note records the evidence-backed SHERA ML program state after the completed
Lonestar6 S01/S05 training runs and the completed Gattaca2 V4 production
render. It is a dated snapshot, not a replacement for the living roadmap in
`docs/dev/shera_ml_inverse_model_design.md`.

## Evidence Classes

Completed results:

- S01 `S01-E01` three-seed baseline validation results on the frozen V3
  benchmark.
- S01 `S01-E02` through `S01-E07` seed-11 optimizer/training-control validation
  results on the same frozen V3 benchmark.
- S05 Wave 1 seed-11 architecture validation results on the same frozen V3
  benchmark.
- Gattaca2 V4 raw render corpus `shera_ml_master_v4`, with task-summary and
  filesystem audits passing.

Provisional conclusions:

- `S01-E03` is the current optimizer/training-control candidate, not a proven
  globally optimal schedule and not a convergence proof.
- `S05-E04` is the provisional architecture winner, not multi-seed validated
  and not yet combined with the `S01-E03` training prescription.

Historical infrastructure events:

- LS6 job 3418670, `S01-E01-R001`, failed immediately before the later accepted
  scientific runs.
- LS6 job 3419329, `S05-E01-LS6-SMOKE`, was cancelled before execution.

Future or unimplemented capability:

- The V4 prepared-dataset layer does not yet exist in this repository state.
- V4 pair curricula and nuisance-invariance training have not yet been run.
- Ordinary S01/S05 model selection did not evaluate the locked test set
  (`evaluate_test: false`).

## Lonestar6 Completed Runs

The following scientific jobs completed successfully on TACC Lonestar6:

| run | LS6 job | status | elapsed |
| --- | ---: | --- | ---: |
| `S01-E01-R001` | 3418678 | COMPLETED | 00:33:19 |
| `S01-E01-R002` | 3418707 | COMPLETED | 00:30:51 |
| `S01-E01-R003` | 3418708 | COMPLETED | 00:26:23 |
| `S05-E01-R001` | 3419396 | COMPLETED | 00:39:57 |
| `S05-E02-R001` | 3419397 | COMPLETED | 00:31:20 |
| `S05-E03-R001` | 3419398 | COMPLETED | 00:30:14 |
| `S05-E04-R001` | 3419399 | COMPLETED | 00:30:04 |
| `S01-E02-R001` | 3419825 | COMPLETED | 01:35:22 |
| `S01-E03-R001` | 3419826 | COMPLETED | 01:21:38 |
| `S01-E04-R001` | 3419827 | COMPLETED | 01:07:12 |
| `S01-E05-R001` | 3419828 | COMPLETED | 01:08:25 |
| `S01-E06-R001` | 3419829 | COMPLETED | 01:20:40 |
| `S01-E07-R001` | 3419830 | COMPLETED | 01:23:02 |

Every meaningful persistent S01/S05 production run directory contains:

- `run_manifest.json`
- `run_config_resolved.json`
- `history.csv`
- `metrics.json`
- `evaluation_predictions.npz`

Smoke directories can also contain complete artifacts, but smoke metrics remain
infrastructure evidence and are not scientific results.

## S01 Baseline

Zero-correction Fisher RMSE for the S01 validation contract is 250.840.

| run | seed | best validation RMSE | best epoch | max epochs | MSE skill |
| --- | ---: | ---: | ---: | ---: | ---: |
| `S01-E01-R001` | 11 | 72.4681 | 99 | 100 | 0.916535 |
| `S01-E01-R002` | 23 | 75.7559 | 94 | 100 | 0.908790 |
| `S01-E01-R003` | 47 | 74.3278 | 93 | 100 | 0.912197 |

The three-seed mean best validation RMSE is 74.1839, with sample standard
deviation 1.6486. This strongly demonstrates pairwise Fisher-scaled correction
learnability and reproducible improvement over zero correction. It does not
prove convergence: all three best epochs occurred near the end of the
100-epoch budget.

## S01 Optimizer/Training-Control Wave

All runs use seed 11 and the canonical S01 architecture and evaluation
contract.

| run | initial LR | scheduler | epochs completed | best epoch | best validation RMSE | MSE skill |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `S01-E02-R001` | 5e-4 | fixed | 300 | 278 | 59.5046 | 0.943726 |
| `S01-E03-R001` | 1e-3 | fixed | 300 | 273 | 57.8676 | 0.946780 |
| `S01-E04-R001` | 5e-4 | reduce-on-plateau | 242 | 219 | 64.9824 | 0.932888 |
| `S01-E05-R001` | 1e-3 | reduce-on-plateau | 229 | 222 | 67.0941 | 0.928455 |
| `S01-E06-R001` | 5e-4 | cosine | 300 | 278 | 60.3250 | 0.942163 |
| `S01-E07-R001` | 1e-3 | cosine | 300 | 275 | 60.3282 | 0.942157 |

`S01-E03` is the current optimizer/training-control candidate. Relative to the
`S01-E01-R001` seed-11 baseline, it reduces best validation RMSE by 20.1475%.
The fixed 1e-3 learning rate with longer training outperformed the tested
plateau and cosine prescriptions in this seed-11 wave. This is not a
convergence claim: `S01-E03`, `S01-E02`, `S01-E06`, and `S01-E07` all reached
their best checkpoints late in the allowed training interval.

## S05 Wave 1

All ordinary S05 Wave 1 runs use seed 11, the frozen V3 S01 benchmark contract,
and `evaluate_test: false`.

| run | architecture | parameters | best epoch | best validation RMSE | final validation RMSE | MSE skill |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `S05-E01-R001` | baseline `concat_diff` | ~767k | 99 | 72.3262 | not recorded here | 0.916862 |
| `S05-E02-R001` | difference-only comparator | not recorded here | 99 | 89.2171 | not recorded here | 0.873496 |
| `S05-E03-R001` | smaller `concat_diff` model | not recorded here | 99 | 89.3951 | not recorded here | 0.872991 |
| `S05-E04-R001` | larger `concat_diff` model | ~3.055M | 93 | 68.4548 | 70.5641 | 0.925524 |

`S05-E01` reproduces the `S01-E01` seed-11 baseline closely, giving a useful
internal comparability check. Difference-only comparison and the smaller model
are substantially worse than baseline. The larger `S05-E04` model is the best
Wave 1 architecture result and improves best validation RMSE by 5.3527%
relative to `S05-E01`. It remains provisional because it has one production
seed, used the old 5e-4 / 100-epoch prescription, and has not been combined
with the `S01-E03` optimizer/training-control candidate.

## V4 Production Render

Dataset: `shera_ml_master_v4`

Canonical durable root:

```text
/projects/shera_hpc/data/ml_training/shera_ml_master_v4
```

Renderer source snapshot:

```text
3da21e603c779377b559c9b86182f7150bd33366
```

Gattaca2 Slurm array job 19450239 used:

- array `0-53%32`;
- 54 tasks;
- 20,000 renders per full task;
- 4,960 renders in the final task;
- 4 CPUs per task;
- 4 GB requested per task;
- 2 hour walltime;
- concurrency cap 32.

Every array task completed with `ExitCode 0:0`. Typical full tasks elapsed
approximately 1h20m to 1h28m. The final partial task elapsed approximately
21m26s. Observed MaxRSS was approximately 0.60 GB for ordinary full tasks and
approximately 0.58 GB for the final partial task. The 4 GB value was the
request, not measured usage.

Task-summary audit:

| field | value |
| --- | ---: |
| summary files | 54 |
| task IDs | 0..53 |
| missing task IDs | 0 |
| unexpected task IDs | 0 |
| attempted | 1,064,960 |
| rendered | 1,064,960 |
| skipped valid | 0 |
| invalid existing | 0 |
| failed | 0 |
| accounted complete | 1,064,960 |
| expected render count | 1,064,960 |
| bad tasks | 0 |
| range count | 54 |
| final stop | 1,064,960 |

`accounted_matches_expected` was true, `coverage_exact` was true, and the final
audit result was `V4_TASK_SUMMARY_AUDIT: PASS`.

Filesystem audit:

| family | split | FITS | JSON |
| --- | --- | ---: | ---: |
| `joint_full_v4` | train | 655,360 | 655,360 |
| `joint_full_v4` | validation | 81,920 | 81,920 |
| `joint_full_v4` | test | 81,920 | 81,920 |
| `radial_capture_v4` | train | 163,840 | 163,840 |
| `radial_capture_v4` | validation | 40,960 | 40,960 |
| `radial_capture_v4` | test | 40,960 | 40,960 |
| total | all | 1,064,960 | 1,064,960 |

The FITS/JSON totals match. The measured canonical corpus footprint is 123 GB.
After completion, `/projects/shera_hpc` reported 3.2 TB size, 878 GB used,
2.3 TB available, and 28% utilization.

The raw V4 corpus is frozen and complete. Do not delete, rewrite, or repair it
outside an explicit audited repair task.

## V4 Scientific Structure

The frozen V4 structure contains 106,496 science states, 10 nuisance states,
and 1,064,960 full cross-product renders. Every science state is rendered
against every fixed nuisance state.

| family | train science | validation science | test science |
| --- | ---: | ---: | ---: |
| `joint_full_v4` | 65,536 | 8,192 | 8,192 |
| `radial_capture_v4` | 16,384 | 4,096 | 4,096 |

This complete crossing enables future controlled pair families:

- same nuisance, different science;
- different nuisance, same science;
- different nuisance, different science;
- identity pairs.

These pair families are future training-design opportunities, not completed
studies. Do not exhaustively materialize all combinatorial pairs. Prefer
dynamic pair generation for training and frozen deterministic pair manifests
for validation and test.

Likely complementary family roles:

- `joint_full_v4`: broad multivariate science-state coverage and the general
  high-dimensional training distribution.
- `radial_capture_v4`: controlled Fisher-distance/capture-radius coverage,
  distance-balanced evaluation, curriculum design, and capture-range
  diagnostics.

Preserve family and split metadata; do not collapse the two families into one
undifferentiated population.

## Recommended Next Work

First bridge experiment on frozen V3:

1. Test the `S05-E04` larger `concat_diff` architecture with the `S01-E03`
   fixed-LR 1e-3 longer-training prescription at seed 11.
2. If clearly promising, repeat only that combined prescription at seeds 23
   and 47.
3. Keep the frozen test set locked during model selection.

Next major implementation task:

- build a V4 prepared-dataset layer that preserves the canonical raw FITS and
  JSON corpus as authoritative;
- derive efficient training arrays or shards;
- preserve `science_state_id`, `nuisance_state_id`, `render_state_id`, family,
  and split role;
- preserve ordered physical and Fisher-scaled vectors;
- remain reproducible and content-addressable;
- support dynamic pair generation and pair-family mixtures;
- support grouped science-state splits without leakage;
- create deterministic frozen validation/test pair manifests;
- preserve a path for V3 regression evaluation;
- reuse or generalize the existing prepared-dataset infrastructure where
  practical.

Open V4 nuisance-generalization question:

- train on all ten nuisance states and evaluate robustness over cross-nuisance
  combinations; or
- reserve one or two nuisance identities globally during development to test
  generalization to nuisance states not observed in training.

Resolve this before freezing the V4 split/evaluation contract. A final
production model may eventually train on all ten nuisance states after
development choices are frozen.

Evaluation axes for the next phase:

- science-state generalization;
- Fisher-distance/capture radius;
- nuisance robustness;
- pair-family behavior.

Continue reporting per-parameter Fisher-scaled metrics, overall Fisher RMSE,
MSE skill, correction-vector geometry, and distance-bin diagnostics. The
existing reusable campaign analyzer in `work/experiments/ml/analysis/` already
supports normalized campaign tables, prediction geometry, per-parameter
metrics, slices, and distance bins; extend that convention rather than creating
a second analysis framework.
