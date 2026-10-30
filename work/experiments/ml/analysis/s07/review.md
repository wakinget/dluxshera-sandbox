# S07 — V4 joint-state generalization: scientific review

**Review date:** 2026-10-30
**Status:** Data complete (10/10 runs, validation only); review complete. No locked-test claim.

## Motivation and hypotheses

S06 established a clean V3 science-correction bridge: a larger `concat_diff` CNN
beat the original on matched seeds, and long training mattered. S07 carries those
findings onto the substantially harder V4 `joint_full_v4` joint-state population,
where every science parameter can move at once. The [study prescription](../../s07/README.md)
and [`study.yaml`](../../s07/study.yaml) define two controlled questions:

1. **Architecture.** Does the larger CNN (~3,055,060 parameters) improve
   generalization on V4 joint-state correction over the original smaller CNN
   (~767,220 parameters)? Planned comparison: E01 versus E02, matched seeds.
2. **Training diversity.** How does restricting the eligible joint-state training
   population to progressively larger nested prefixes affect generalization?
   Planned comparison: E03 → E06 at seed 11.

Everything below the architecture and prefix comparisons — residual subspace
structure, matched-mirror correlations, reverse-pair antisymmetry — is a *post-hoc*
diagnostic, not a pre-registered test.

## Design and controls

| Experiment | Model | Eligible joint-state training population | Seeds |
| --- | --- | --- | --- |
| S07-E01 | Original CNN, 767,220 params | unrestricted | 11, 23, 47 |
| S07-E02 | Larger CNN, 3,055,060 params | unrestricted | 11, 23, 47 |
| S07-E03 | Larger CNN | 4,096-record prefix | 11 |
| S07-E04 | Larger CNN | 8,192-record prefix | 11 |
| S07-E05 | Larger CNN | 16,384-record prefix | 11 |
| S07-E06 | Larger CNN | 32,768-record prefix | 11 |

Verified against each `run_config_resolved.json`: all ten runs share AdamW,
learning rate `1e-3`, weight decay `1e-4`, batch size 32, 16,384 **ordered** pairs
per epoch, a nominal 600-epoch budget, no learning-rate schedule, no injected
observation noise, and `include_reverse: true`. Early stopping monitors validation
loss with `min_epochs: 120`, `patience: 80`, `min_delta_relative: 5e-4`.
Model parameter counts in the run table match the stated values exactly.
Architecture is the *only* difference between E01 and E02; the model is held fixed
across E02–E06 and only `joint_train_prefix_size` varies.

### What `joint_train_prefix_size` actually restricts

`joint_train_prefix_size` filters the eligible **training science states** of the
`joint_full_v4` family by `joint_train_sequence_index < prefix`
(`src/dluxshera/ml/pairs.py`). It does not change pairs per epoch, the
validation population, or the number of distinct physical configurations in any
other family. The prefixes are the frozen nested V4 train prefixes
(`V4_TRAIN_PREFIXES`, `src/dluxshera/datasets/prepared_v4.py`), drawn from the
65,536 `joint_full_v4` train science states and interleaved so each prefix
preserves an exact 1:1:1:1 scale-stratum mixture
([spec](../../datasets/master_v4_spec.md)).

Two consequences matter for interpretation:

- **A prefix record is a science state, not an independent physical
  configuration in the colloquial sense.** Each state is rendered against all ten
  fixed nuisance states, so the available *image* population is ten times the
  prefix, and the nuisance bank is identical in every experiment. E03's deficit is
  a restriction on *science-state diversity*, not on sample count per epoch.
- **Even E06 is a restriction.** 32,768 is half of the 65,536 available train
  states, so E06 is not the unrestricted condition — E01/E02 are.

## Provenance and completeness

| Item | Value |
| --- | --- |
| Cluster / source root | LS6 (`ls6.tacc.utexas.edu`), `/scratch/11689/dmckeith/dLuxShera-ML` |
| Active import namespace | `Results/hpc_imports/ml/ls6/S07/` |
| Export produced | 2026-10-30T17:48:02Z |
| Archive | `s07_complete_20261030T174759Z.tar.gz`, SHA256 `5937e7e6a739fdab2214745ea9126511e91c085f82511c81932277585f392aea` |
| Source revision (all ten runs) | `5fa65027f1fe4c7495f9be292303d25fce82fe4e` |
| Working tree | **dirty** in every run manifest |
| Runs expected / observed | 10 / 10 |
| Test evaluated | **false** in all ten manifests |

All 50 compact artifacts were hashed and matched byte-for-byte against the
`s07_export_manifest.json` SHA-256 inventory, and again against the installed
import's own `files_sha256`: zero mismatches. The loader reports no warnings.

**Dirty-tree limitation.** All ten runs record `git.dirty = true`. The exact
working-tree diff is not recoverable from the compact artifacts, so the recorded
revision identifies the commit the runs were launched from but does not fully
pin the executed code. Three runs (E03/E04/E05) have `git.source_commit = null`
while `git.commit` carries the revision; the loader's source-commit fallback
resolves these consistently.

**Import reconciliation.** The pre-existing local S07 import (synced
2026-09-29T01:23:55Z) was an **in-flight snapshot of the same ten runs at the
same revision**, not a different experiment generation. Four runs disagreed
(E01-R002, E02-R001, E02-R003, E06-R001), each because the older copy was
captured mid-training with 316/360/358/360 history rows against the completed
600/600/591/600; the other six were byte-identical. It is preserved outside active
discovery at `Results/hpc_imports/ml_history/ls6/S07_inflight_20260929/`.
See [reconciliation](reconciliation.md) for the full comparison.

### Stopping reasons — no incomplete runs

Six runs reached the full 600-epoch budget; four stopped early under the
configured early-stopping rule. Early stopping is a legitimate finish, not a
failure, and `best_epoch` is generally *not* the last epoch.

| Run | Experiment | Seed | Epochs completed | Last epoch | Best epoch | Stopping reason |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| E01-R001 | S07-E01 | 11 | 600 | 599 | 567 | reached budget |
| E01-R002 | S07-E01 | 23 | 600 | 599 | 598 | reached budget |
| E01-R003 | S07-E01 | 47 | 600 | 599 | 569 | reached budget |
| E02-R001 | S07-E02 | 11 | 600 | 599 | 598 | reached budget |
| E02-R002 | S07-E02 | 23 | 600 | 599 | 576 | reached budget |
| E02-R003 | S07-E02 | 47 | 591 | 590 | 510 | early stopped |
| E03-R001 | S07-E03 | 11 | 199 | 198 | 20 | early stopped |
| E04-R001 | S07-E04 | 11 | 199 | 198 | 42 | early stopped |
| E05-R001 | S07-E05 | 11 | 199 | 198 | 102 | early stopped |
| E06-R001 | S07-E06 | 11 | 600 | 599 | 593 | reached budget |

All headline RMSEs below are **best-checkpoint** validation values, matching
`metrics.json`; final-epoch values are worse in every run and are reported
separately in `tables/train_validation_divergence.csv`.

## Evaluation contract

Every run shares one evaluation contract, confirmed from the run manifests and
again from the saved arrays:

| Identity | Value |
| --- | --- |
| Prepared dataset (PREP-V4-v1) | `2b6defd272a3ecf7c28bae15d396e29236644d8e1ff44251815314f54bb98e7f` |
| Split registry (SPLIT-V4-ROLE-PRESERVING-v1) | `951f9d38c3a33a2c5d24e7c63df7938b24a44d8c9945d4f3be7d934d7fa5467a` |
| Validation artifact | `S07-VALIDATION-PAIRS-v1`, content SHA256 `27608006dd9d76e3eb59a4eca9bbfafe04d7c50880b4d8ae949fbf577f534505` |
| Scaler | `SCALER-V4-GLOBAL-MAX-ABS-v1` |
| Evaluation slice | `heldout_science_seen_nuisance` only |
| Ordered records | 1,024 |
| Zero-correction Fisher RMSE | 386.623249 (identical for all ten runs) |

Taking E01-R001 as reference, all ten runs have **identical** `pair_record_id`
arrays in identical order, `y_true_z` max absolute difference 0.0, and
`fisher_distance_l2` max absolute difference 0.0. Only `y_pred_z` differs. Paired
comparisons are therefore exact per-record comparisons
(`tables/array_consistency.csv`).

### Parameter order

The saved array column order is

```text
[separation_as, log_flux_total, contrast, plate_scale, M1 Z4-Z11, M2 Z4-Z11]
```

recovered by matching each archived per-parameter RMSE to the column-wise RMSE
recomputed from the saved arrays, and consistent with S06's independently
documented order. This matters: `metrics.json` is serialized with
`sort_keys=True`, so its `fisher_per_parameter_rmse` key order is **alphabetical
and is not the array column order**. A prior reading of these results used the
sorted keys positionally, which shifts the optical block by three columns and
mixes M1 Z7–Z11 with M2 Z4–Z6 and the source parameters. `campaign_analysis.py`
now performs the value-based recovery and falls back to positional `z[i]` labels
rather than guessing when the match is ambiguous.

### Comparability limits

- The evaluation slice is `heldout_science_seen_nuisance`: **science states are
  held out, nuisance states are not.** S07 does not measure generalization to an
  unseen registration-nuisance population. There is no held-out-nuisance slice
  in this study to compare against.
- The 1,024 ordered records are **512 science pairs plus their exact reversals**,
  confirmed directly: every record has exactly one partner with `t_j = -t_i`, 512
  independent pairs, zero unpaired records. Treating the records as independent
  understates the standard error by a factor of ≈1.414 in every run
  (`tables/pair_clustered_uncertainty.csv`). Quoted seed scatter below is a
  3-seed sample SD, which is the binding uncertainty here, not the within-run SE.
- **V3 and V4 absolute RMSE are not comparable.** The zero-correction baseline is
  250.84 in S06 versus 386.623 here, on a different dataset, scaler and pairing
  policy. Compare skill and within-study contrasts, not absolute RMSE.

## Quantitative findings

All metrics were recomputed from the saved prediction arrays using
`campaign_analysis.compute_initializer_metrics` and compared with `metrics.json`:
**maximum absolute RMSE difference 0.0, maximum absolute skill difference 0.0**
across all ten runs (`tables/recomputed_vs_archived_metrics.csv`).

### Architecture: the larger model wins consistently

| Experiment | Mean best-validation Fisher RMSE | Sample SD | Mean MSE skill |
| --- | ---: | ---: | ---: |
| S07-E01 (original CNN) | 139.604 | 2.482 | 0.8696 |
| **S07-E02 (larger CNN)** | **127.722** | **1.807** | **0.8909** |

| Seed | E01 | E02 | Delta |
| ---: | ---: | ---: | ---: |
| 11 | 137.097 | 129.745 | −7.352 |
| 23 | 142.060 | 126.268 | −15.793 |
| 47 | 139.655 | 127.152 | −12.502 |

The larger model improves the mean by **8.51%** and wins on **all three matched
seeds** (`classify_metric_deltas` → "all improved"). The improvement exceeds the
seed scatter of either experiment, so at three seeds this is a consistent
direction, not a coin flip — though three seeds still cannot bound the effect
tightly.

**The advantage is not an artifact of unequal training duration.** E02-R003 early
stopped at 591 epochs while all E01 runs ran the full 600. Comparing both at a
common epoch (0-based index 300) still favours the larger model: mean validation
RMSE **155.86 (E01)** versus **145.88 (E02)**. (At the 1-based `epoch_number` 300
convention the values are 154.98 and 145.94; both conventions are tabulated in
`tables/validation_cutoffs.csv`, and they differ by one sampled epoch.)

### Training population: restriction hurts, and the smallest prefixes overfit

| Experiment | Prefix (train science states) | Best RMSE | Best epoch | Last epoch | Stopping |
| --- | ---: | ---: | ---: | ---: | --- |
| S07-E03 | 4,096 | 172.649 | 20 | 198 | early stopped |
| S07-E04 | 8,192 | 167.968 | 42 | 198 | early stopped |
| S07-E05 | 16,384 | 165.337 | 102 | 198 | early stopped |
| S07-E06 | 32,768 | 136.582 | 593 | 599 | reached budget |

The best epoch rises monotonically with prefix size (20 → 42 → 102 → 593): the
smaller the training population, the sooner the model stops improving on
validation. The train/validation histories make the mechanism explicit
(`tables/train_validation_divergence.csv`, `figures/prefix_train_vs_validation_loss.png`):

| Run | Validation RMSE change, best → last | Train loss change, best → last |
| --- | ---: | ---: |
| E03-R001 | **+18.6% worse** | 81.1% lower |
| E04-R001 | **+11.1% worse** | 62.0% lower |
| E05-R001 | **+1.5% worse** | 31.5% lower |
| E06-R001 | +0.5% worse | 3.5% *higher* |

E03 drives its training loss down by a factor of five after its best epoch while
validation deteriorates by nearly a fifth. That is classical overfitting to a
restricted science-state population, and the effect weakens monotonically as the
prefix grows. E06 shows no such divergence.

**Do not attribute the whole E05→E06 gap (165.34 → 136.58) to population size.**
E03–E05 all early stopped at 199 epochs while E06 completed 600, so that
comparison confounds population size with completed optimization duration. Two
observations bound the confound without removing it: at a common epoch-300 cutoff
E06 is already ahead (151.72 versus E05's 167.87, which is E05's last available
epoch), and E05's own validation curve had flattened and triggered early stopping
rather than being cut off while still improving. The direction is well supported;
the magnitude is not cleanly decomposed.

Finally, **E06 at 136.58 remains worse than E02's 129.75 at the same seed 11**,
consistent with 32,768 still being half the available train states.

### Longer training remains useful for the unrestricted models

Best epochs for the six unrestricted runs are 567, 598, 569, 598, 576 and 510 out
of a 600 budget — most best checkpoints occur in the last sixth of training. The
600-epoch schedule should not be described as converged for E01/E02/E06.

### Validation performance (E02)

Against the common zero-correction Fisher RMSE of **386.623**:

| Quantity | E02 mean |
| --- | ---: |
| Best validation Fisher RMSE | 127.722 |
| MSE skill | 0.8909 |
| Fraction of ordered pairs improved by one correction | **100%** |
| Mean initial correction distance | 1,556.00 |
| Mean remaining distance | 482.97 |
| Physical binary-separation residual RMSE | **4.09 mas** |

"Fraction improved" is `relative_residual_norm < 1`, i.e. the learned correction
moved the state closer than doing nothing in Fisher-scaled Euclidean distance.
It holds for all 1,024 ordered records in all three E02 seeds. This is a
**single-step synthetic validation** result on noiseless renders; it is not a
demonstrated ADORA capture range.

### Optical residual structure (E02)

Shares of total residual squared error, in the saved scaled-z coordinates:

| Block | Share |
| --- | ---: |
| M1 Zernike coordinates | 44.5% |
| M2 Zernike coordinates | 55.1% |
| Source parameters and plate scale | 0.5% |

The residual error is almost entirely optical. Within the optical block, **88.8%**
of the squared error lies in the eight equal-weight M1-minus-M2 difference
coordinates `(e_M1,Zi − e_M2,Zi)/sqrt(2)`, and the first five principal components
of the centered optical residual covariance explain **89.9%** of its variance.

Matched M1/M2 residual correlations, averaged over the three E02 seeds, for
Z4–Z11:

```text
[-0.413, -0.979, -0.981, -0.387, -0.399, -0.990, -0.988, -0.957]
```

These are **learned residual** correlations. The corresponding **injected target**
correlations in the same coordinates are near zero
(+0.014, −0.027, +0.014, +0.093, +0.028, −0.094, +0.066, −0.076), so the strong
anti-correlation is a property of the trained model's errors and is not inherited
from paired target sampling (`figures/e02_matched_mirror_residual_correlations.png`).

The structure is not purely diagonal. The only off-diagonal pairs exceeding 0.3 in
magnitude are **M1 Z4 × M2 Z11 (+0.597)** and **M1 Z11 × M2 Z4 (−0.442)** — a
Z4/Z11 cross-mirror coupling that echoes S06's Schur-FIM finding that Z11 is the
exception to one-to-one mirror matching (its same-index Schur cross-correlation is
only +0.280, while M1 Z4 and M2 Z11 show ≈−0.960 cross sensitivity).

**Relation to the S06 Schur analysis — and what not to conclude.** S06's
registration-nuisance-marginalized Schur FIM found matched M1/M2 Z4–Z10 normalized
cross sensitivities of 0.99991–0.99997, i.e. the two mirrors produce nearly
indistinguishable local image perturbations. That independently supports reading
S07's residual anti-correlation as a mirror-assignment identifiability problem.
But the simple equal-weight difference subspace **is not** the physical Fisher
eigenbasis: S06's weak physical eigenvectors are asymmetric (representative
loadings near (+0.472 M1, −0.881 M2) for Z4–Z6 and (+0.365 M1, −0.931 M2) for
Z7–Z10) and mix several Zernikes, with mean squared projection onto the
equal-weight difference span of about 82.5%. The S06 product is also a **V3**
flight-simple reference, so it is qualitative support for S07's V4 structure, not a
validated coordinate-space projection. The scaled-z → physical-theta transform
remains an open question and is not used here.

### Distance dependence and pair consistency

E02 means by initial Fisher-distance bin (bin populations are identical across all
ten runs):

| Initial Fisher distance | Ordered records | MSE skill | Model RMSE | Baseline RMSE | Median relative residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–100 | 0 | — | — | — | — |
| 100–250 | **4** | 0.892 | 18.13 | 55.41 | 0.311 |
| 250–500 | 60 | 0.912 | 26.30 | 88.66 | 0.264 |
| 500–1,000 | 226 | 0.918 | 48.36 | 168.43 | 0.256 |
| 1,000–2,000 | 384 | 0.898 | 106.63 | 333.39 | 0.286 |
| 2,000–5,000 | 350 | 0.886 | 183.35 | 543.77 | 0.325 |

The 0–100 bin is empty. The **100–250 bin holds only four ordered records, which
are two science pairs and their reversals** — effectively two independent samples.
Its skill value carries essentially no statistical weight and should not be read
as capture-range evidence at small displacement. Skill is roughly flat at 0.89–0.92
across populated bins; absolute residual RMSE grows with initial displacement even
where fractional skill does not fall.

**Prediction antisymmetry.** Because the validation set contains exact reverse
pairs, a self-consistent model should satisfy `p(B,A) = −p(A,B)`. Measured
antisymmetry violation `RMS(p_AB + p_BA)` is small but nonzero: 2.3–2.7% of
prediction RMS for E01, 2.3–2.7% for E02, and 1.5–2.9% across the prefix runs
(`tables/reverse_pair_antisymmetry.csv`). Nothing in S07 enforces this symmetry,
so the residual violation is a free diagnostic and a useful baseline for S11's
consistency-loss experiments — it is a small effect relative to the ≈33% relative
residual norm, so a consistency loss should not be expected to recover much error
on its own.

## Interpretation and limitations

### Supported by these measurements

1. **Larger model capacity consistently helps on the V4 joint-state task.**
   8.51% mean RMSE improvement, all three matched seeds improved, and the
   advantage persists at a common epoch count.
2. **Restricting the training science-state population substantially degrades
   generalization**, monotonically in prefix size, with explicit validation
   deterioration and train/validation divergence for the smallest prefixes.
3. **Longer training remains useful for the unrestricted models**; most best
   checkpoints fall in the final sixth of a 600-epoch budget.
4. **The remaining error is dominated by structured optical-mode ambiguity.**
   99.5% of residual squared error is optical, 88.8% of the optical part lies in
   equal-weight M1−M2 difference coordinates, and matched-mirror residual
   correlations reach −0.99 where the injected targets are uncorrelated.
5. **Learned single-step corrections substantially reduce synthetic
   validation-state error**: skill 0.891, 100% of ordered pairs improved, mean
   Fisher distance 1,556 → 483, separation residual 4.09 mas.

### Not demonstrated

- **Generalization to unseen registration-nuisance populations.** The only
  evaluation slice is `heldout_science_seen_nuisance`.
- **Robustness to observation noise or a mismatched forward model.** Noise was
  disabled in every run and the forward model is the training model.
- **Microarcsecond-level astrometric recovery.** The separation residual is
  4.09 mas.
- **End-to-end improvement in ADORA convergence.** Single-step Fisher-distance
  reduction is not a capture-range or convergence result.
- **Any locked-test performance.** `test_evaluated = false` in all ten manifests;
  no test evaluation was performed in this review.
- **Saturation of capacity, training duration, or dataset size.** All three
  were still improving at the edge of what S07 explored: E02 beats E01 at 4×
  parameters, best checkpoints arrive late, and E02 (unrestricted) still beats
  E06 (half the train states).
- **Equivalence of the scaled-coordinate mirror-difference subspace with the
  true physical Fisher eigenvectors**, which are asymmetric and cross-coupled.

Absolute RMSE values here are not comparable with S06's V3 numbers.

## Reproducibility

Regenerate every table, figure and headline number from the imported artifacts:

```bash
.venv/bin/python Results/ml_analysis/S07/reproduce_s07.py
```

That script is a thin local driver over the maintained helpers
(`load_campaign`, `experiment_summary_table`, `matched_seed_delta_table`,
`validation_cutoff_table`, `residual_error_budget_table`, `campaign_plots`);
only the genuinely S07-specific diagnostics — optical residual subspace, matched
M1/M2 correlations, reverse-pair antisymmetry and pair-clustered uncertainty —
are computed in it, directly from the saved scaled-z arrays. It reads
`Results/hpc_imports/ml/ls6/S07/` and writes to `Results/ml_analysis/S07/`.
It asserts 10 runs and `test_evaluated == false`, and performs no inference,
no test evaluation and no GPU work.

Generated locally (all paths under `Results/ml_analysis/S07/`, git-ignored):

- `headline_summary.json` — every number quoted above.
- `tables/` — `runs.csv`, `history.csv`, `parameters.csv`, `slices.csv`,
  `distance_bins.csv`, `evaluation_contract.csv`, `experiment_summary.csv`,
  `matched_seed_deltas_e01_e02.csv`, `recomputed_vs_archived_metrics.csv`,
  `array_consistency.csv`, `validation_cutoffs.csv`,
  `train_validation_divergence.csv`, `reverse_pair_antisymmetry.csv`,
  `pair_clustered_uncertainty.csv`, `optical_residual_shares.csv`,
  `matched_mirror_residual_correlations.csv`, `optical_residual_pca.csv`,
  `residual_error_budget.csv`, `training_prefix_sweep.csv`.
- `figures/` — `experiment_best_rmse.png`, `matched_seed_deltas_e01_e02.png`,
  `validation_history_by_experiment.png`, `best_so_far_rmse.png`,
  `prefix_train_vs_validation_loss.png`, `training_prefix_sweep.png`,
  `parameter_skill_heatmap.png`, `e02_parameter_error_budget.png`,
  `e02_matched_mirror_residual_correlations.png`,
  `mse_skill_by_distance_bin.png`.

## Follow-up and tracking

**Open identity questions.** The dirty working tree in all ten runs leaves the
executed code not fully pinned. The scaled-z → physical-theta transform is still
unvalidated, so no residual projection into the S06 Schur eigenbasis is supported.

**Proposed controlled follow-ups.**

1. **Equal-duration prefix sweep.** Rerun E03–E06 with early stopping relaxed (or
   a fixed 600-epoch budget with best-checkpoint selection) to separate
   population size from completed optimization duration.
2. **Complete the prefix ladder to 65,536.** E06 at 32,768 is still half the
   available train states, and E02 beats it; the 65,536 point would show whether
   dataset size has saturated.
3. **Capacity ladder.** E02 beats E01 at 4× parameters with no sign of
   saturation; a third, larger model at matched seeds would locate the knee.
4. **Held-out-nuisance evaluation slice for V4.** S07 cannot speak to nuisance
   generalization at all. Adding a `heldout_science_heldout_nuisance` slice to the
   V4 evaluation contract would close the largest interpretive gap, without
   touching the existing frozen validation artifact.
5. **Noise and forward-model mismatch sensitivity**, before any downstream
   ADORA claim.
6. **S11 consistency loss.** The measured 2–3% antisymmetry violation is the
   natural pre-treatment baseline; S07's reverse-pair table supplies it per run.

Do not tune against this frozen validation artifact. The locked test set remains
unevaluated and should stay that way until model selection is final.

**Tracking.** [ML study index and workflow](../README.md) ·
[campaign tracker notebook](../ml_campaign_tracker.ipynb) ·
[S07 provenance reconciliation](reconciliation.md) ·
[S06 review](../s06/review.md) (completed example and Schur reference) ·
[study prescription](../../s07/README.md).
