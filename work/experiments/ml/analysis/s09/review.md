# S09 — V4 distance-balanced radial capture: scientific review

**Review date:** 2026-10-30
**Status:** Data complete (9/9 runs at budget, validation only); review complete.
No locked-test claim.

## Motivation and hypotheses

S07 established the larger `concat_diff` CNN as the V4 reference and showed that
its residual error on `joint_full_v4` joint states is almost entirely structured
optical-mode ambiguity. It did not probe the *range* of correction the network can
handle: its training pairs were drawn from one science-state population, and its
validation Fisher-distance bins below 250 were nearly empty.

S09 asks whether deliberately enriching training with large-displacement pairs
improves the learned initializer. The V4 master dataset carries a second science
family, `radial_capture_v4`, materialized specifically to populate radial
displacement. The [study prescription](../../s09/README.md) and
[`study.yaml`](../../s09/study.yaml) define two planned comparisons:

1. **Radial mixture ratio.** Does raising the configured `radial_capture_v4`
   sampling weight from 25% to 50% improve correction quality? Planned comparison:
   E01 versus E02, matched seeds.
2. **Distance curriculum.** Does easy-to-hard ordering of radial distance bins
   improve on uniform distance-balanced sampling? Planned comparison: E02 versus
   E03, matched seeds.

Everything else below — family-stratified metrics, the mirror-difference subspace,
reversal-coupling accounting, the curriculum exposure decomposition — is a
*post-hoc* diagnostic, not a pre-registered test.

## Design and controls

| Experiment | Pair policy | Configured family mixture | Radial distance-bin weights | Seeds |
| --- | --- | --- | --- | ---: |
| S09-E01 | `s09_v4_75_joint_25_radial` | 75% joint / 25% radial | uniform over all six bins | 11, 23, 47 |
| **S09-E02** | `s09_v4_50_joint_50_radial` | 50% joint / 50% radial | uniform over all six bins | 11, 23, 47 |
| S09-E03 | `s09_v4_curriculum` | 50% joint / 50% radial | staged (see below) | 11, 23, 47 |

Verified from each `run_config_resolved.json` and `run_manifest.json`: all nine
runs share the **larger CNN** (3,055,060 parameters, `concat_diff` comparator,
batch normalization, channels `[32, 64, 128, 256]`), AdamW, learning rate `1e-3`,
weight decay `1e-4`, batch size 32, **16,384 ordered pairs per epoch**
(8,192 base pairs with `include_reverse: true`), a 600-epoch budget, no
learning-rate schedule, and no injected observation noise. Early stopping was
enabled (`min_epochs: 120`, `patience: 80`, `min_delta_relative: 5e-4`) but
**never triggered**: `early_stopped = false` and `reached_max_epochs = true` in all
nine runs. The pair policy is the only difference between experiments.

Mean wall-clock per run was 45.6 h (E01), 46.4 h (E02) and 43.3 h (E03).

### Pair-generation semantics

All S09 pairs are nuisance pair-family **A** (`same_nuisance_different_science`),
with `family_weights: {A: 1.0}`, `same_pair_id: false`, `include_reverse: true` and
`require_same_dataset_family: true`. The ordered-pair convention recorded in every
manifest is `A = current/reference/model`, `B = target/observation`, with
`target_delta_z = z_B − z_A`.

Three distinctions matter for reading the results, and each has been a source of
confusion:

- **Dataset families are a different vocabulary from pair families.**
  `joint_full_v4` and `radial_capture_v4` are *science-state populations* in the V4
  master dataset. `A`/`B`/`C`/`I` are *nuisance pair families* (same-nuisance,
  same-science, cross, identity). S09 uses dataset-family mixing within a single
  pair family; it is not a pair-family study. `require_same_dataset_family: true`
  means both members of a pair come from the same science population, so no pair
  straddles the two.
- **The Fisher distance is a property of the pair, not of either state.** The
  binned quantity `fisher_distance_l2` is `‖z_B − z_A‖` — the norm of the pairwise
  target correction. It is *not* the radius of either science state from nominal.
  A `radial_capture_v4` state at large radius can pair with another large-radius
  state and yield a small correction, and two `joint_full_v4` states can be far
  apart. Distance bins therefore measure how large a correction the network is
  asked for, not how extreme the underlying physical state is.
- **Mixture weights are configured sampling probabilities, not accepted
  proportions.** `dataset_family_sampling` weights and `distance_bin_weights` set
  draw probabilities; acceptance is then subject to `require_same_dataset_family`,
  the distance-bin edges and `max_sampling_attempts: 8000`. The compact artifacts
  do **not** record realized accepted counts per family or bin, so no claim is made
  here about the proportions actually trained on. The evaluation artifact shows
  directly that configured and accepted proportions diverge substantially (below).

### E03 curriculum

`curriculum_stages` restricts which radial distance bins are eligible, by epoch:

| Stage | Epochs | Eligible radial bins |
| ---: | --- | --- |
| 1 | 0–99 | 0–100, 100–250, 250–500 |
| 2 | 100–249 | 0–100 … 1,000–2,000 |
| 3 | 250–599 | all six bins, through 5,000 |

The joint/radial split stays at 50/50 throughout; only the radial bin support
changes.

## Provenance and completeness

| Item | Value |
| --- | --- |
| Cluster / source root | LS6 (`ls6.tacc.utexas.edu`), `/scratch/11689/dmckeith/dLuxShera-ML` |
| Active import namespace | `Results/hpc_imports/ml/ls6/S09/` |
| Export produced | 2026-10-30T20:15:19Z |
| Archive | `s09_complete_20261030T201514Z.tar.gz`, SHA256 `4458733e1f700ef5e8d5407904db8a97e00c0c74755ed4060c4ed706d767af51` (1,865,114 bytes) |
| Source revision (all nine runs) | `5fa65027f1fe4c7495f9be292303d25fce82fe4e` |
| Working tree | **dirty** in every run manifest |
| Runs expected / observed | 9 / 9 |
| Test evaluated | **false** in all nine manifests |

All 45 compact artifacts (five per run) were hashed against the
`s09_export_manifest.json` SHA-256 inventory: **45 verified, 0 missing, 0
mismatched**. The tar contains no absolute paths, no `..` components and no
non-regular members. The loader reports no warnings.

**Import reconciliation — no change required.** All 45 exported artifacts are
**byte-identical** to the pre-existing local import (synced 2026-09-28T22:47:46Z).
The completed export and the installed collection are the same data, so under the
conservative snapshot policy nothing was copied and nothing was overwritten. The
only file present locally but absent from the export is the import's own
`sync_manifest.json`, which is local provenance rather than run data. The archive
and its extracted snapshot are retained at
`Results/ml_analysis/S09/s09_complete_20261030T201514Z/` as provenance.

This is a different case from S07, where the pre-existing copy was an in-flight
mid-training snapshot of the same runs and had to be superseded. Here the earlier
sync already captured the finished runs.

**Dirty-tree limitation.** All nine runs record `git.dirty = true`. The working-tree
diff is not recoverable from the compact artifacts, so the recorded revision
identifies the commit the runs were launched from but does not fully pin the
executed code.

### Stopping state — all nine at budget

Every run completed 600 epochs (`last_epoch = 599`), with `metrics_finalized` and
a best checkpoint present. Best epochs:

| Experiment | Seed 11 | Seed 23 | Seed 47 |
| --- | ---: | ---: | ---: |
| S09-E01 | 594 | 588 | 596 |
| S09-E02 | 596 | 594 | 573 |
| S09-E03 | 596 | 595 | 599 |

All nine best checkpoints fall in epochs **573–599**, the last 5% of the budget.
The 600-epoch schedule bounds these results; it should **not** be described as
converged. Headline numbers below are best-checkpoint validation values matching
`metrics.json`.

## Evaluation contract

All nine runs share one contract, confirmed from the manifests and again from the
saved arrays:

| Identity | Value |
| --- | --- |
| Prepared dataset (PREP-V4-v1) | `2b6defd272a3ecf7c28bae15d396e29236644d8e1ff44251815314f54bb98e7f` |
| Split registry (SPLIT-V4-ROLE-PRESERVING-v1) | `951f9d38c3a33a2c5d24e7c63df7938b24a44d8c9945d4f3be7d934d7fa5467a` |
| Validation artifact | `S09-VALIDATION-PAIRS-v1`, content SHA256 `afab80236eb72b3804366e0f558f4a85948b036e1a078ea9d4f862b69872bad9` |
| Scaler | `SCALER-V4-GLOBAL-MAX-ABS-v1` |
| Evaluation slice | `radial_validation_seen_nuisance` only |
| Ordered records | 1,024 |
| Zero-correction Fisher RMSE | 367.017131 (identical for all nine runs) |

The validation artifact was generated under the E02 policy
(`s09_v4_50_joint_50_radial`, seed 9001), so **E02's training mixture matches the
validation generator's nominal mixture while E01's and E03's do not**. That is a
design asymmetry worth naming, though the accepted composition below shows the
nominal mixture is not what the artifact actually contains.

Pair-record IDs are identical and in identical order across all nine runs, so
matched-seed comparisons are exact per-record comparisons.

### Parameter order — and a corrected defect

The saved array column order is the canonical V4 science vector:

```text
[separation_as, log_flux_total, contrast, plate_scale, M1 Z4-Z11, M2 Z4-Z11]
```

verified against the authoritative contract
(`science_vector_space_identity.ordered_labels` in the V4 `vector_spaces.json`,
science vector-space ID `916a005234f944569f9e635873da6b79a6df505e26fcfd002ffb389cdec19f1b`,
aggregate contract hash `732770f2…2d2b165`) and consistent with
[`master_v4_spec.md`](../../datasets/master_v4_spec.md).

`metrics.json` is serialized with `sort_keys=True`, so its
`fisher_per_parameter_rmse` key order is **alphabetical and is not the array column
order**. Reading those keys positionally shifts the optical block by three columns
and mixes M1 Z7–Z11 with M2 Z4–Z6 and the source parameters. The analysis layer now
recovers the true order by rank-matching each archived per-parameter RMSE against
the column-wise RMSE recomputed from the saved arrays, and falls back to positional
`z[i]` labels rather than guessing when the match is genuinely ambiguous.

Two consequences for S09 specifically:

- **An earlier S09 per-parameter interpretation was wrong and is superseded
  below.** In S09-E02-R001 the sorted-key reading assigned Fisher RMSE 196.706 to
  binary separation; that value is the **M2 Z11** coordinate. The archived named
  binary-separation value is 22.581.
- **S09-E01-R003 was additionally dropped from per-parameter aggregates.** Its
  M1 Z6 (123.16131) and M2 Z6 (123.14989) differ by 1.1e-2 absolute, which an
  earlier per-column match with a loose relative tolerance could not separate, so
  the whole run degraded to positional labels, contributed no separation row, and
  silently left E01's per-parameter means computed over two seeds instead of three.
  Rank-matching with a tolerance appropriate to the bit-exact agreement between
  archived and recomputed values resolves it. Regression coverage is in
  `tests/ml/test_campaign_analysis.py`.

**Verification of the corrected mapping.** For all nine runs the per-parameter
values recomputed from the prediction arrays match the named entries in
`metrics.json`, and `sqrt(mean(per-coordinate MSE))` reproduces each run's global
Fisher RMSE to ≤1.4e-14 — confirming the per-parameter rows tile every column
exactly once.

### Comparability limits

- **The slice is `radial_validation_seen_nuisance`: science states are held out,
  nuisance states are not.** S09 says nothing about generalization to an unseen
  registration-nuisance population. There is no held-out-nuisance slice in this
  study.
- **The 1,024 ordered records are 512 reversal-coupled pairs.** Every record has
  exactly one partner with `t_j = −t_i` and the same `fisher_distance_l2`;
  1,024/1,024 records are matched, so there are **512** independent pairs, not
  1,024 samples. Treating records as independent understates the standard error by
  ≈1.414. The binding uncertainty here is the 3-seed sample SD.
- **S09 and S07 absolute RMSE are not comparable.** The zero-correction baseline is
  367.017 here against 386.623 in S07, because the validation population is drawn
  under a different pair policy mixing a second science family. Compare skill and
  within-study contrasts, not absolute RMSE across studies.

## Quantitative findings

### Aggregate results

Against the common zero-correction Fisher RMSE of **367.017**:

| Metric | E01 | **E02** | E03 |
| --- | ---: | ---: | ---: |
| Mean best-validation Fisher RMSE | 116.635 | **114.796** | 116.345 |
| Sample SD (3 seeds) | 0.693 | 0.967 | 1.897 |
| Mean global MSE skill | 0.8990 | **0.9022** | 0.8995 |
| Mean physical separation RMSE (mas) | 4.042 | **3.644** | 3.987 |
| Mean separation Fisher MSE skill | 0.9965 | **0.9972** | 0.9966 |
| Fraction of ordered pairs improved | 1.000 | 1.000 | 1.000 |
| Mean initial Fisher distance | 1,446.4 | 1,446.4 | 1,446.4 |
| Mean remaining Fisher distance | 436.4 | **431.1** | 434.3 |
| Mean correction cosine alignment | 0.9501 | 0.9509 | 0.9513 |

E02 improves mean global Fisher RMSE by **1.58%** and mean physical separation RMSE
by **9.84%** relative to E01.

"Fraction improved" is `relative_residual_norm < 1` — the learned correction moved
the state closer than doing nothing in Fisher-scaled Euclidean distance. It holds
for all 1,024 ordered records in all nine runs. This is a single-step result on
noiseless synthetic renders; it is **not** a demonstrated ADORA capture range.

### Matched-seed comparisons — the two metrics disagree

**E02 minus E01:**

| Seed | Fisher RMSE delta | Separation RMSE delta (mas) |
| ---: | ---: | ---: |
| 11 | −2.337 | −0.384 |
| 23 | **+0.010** | −0.358 |
| 47 | −3.189 | −0.450 |

The **separation improvement is consistent: all three matched seeds improve**, by
0.36–0.45 mas. The **global Fisher RMSE improvement is not consistent** — seed 23
is essentially unchanged (+0.010, i.e. marginally worse), and the two seeds that do
improve do so by more than the E01 seed scatter. So the 1.58% aggregate Fisher
improvement rests on two of three seeds and should be read as suggestive, while the
9.84% separation improvement is directionally solid at three seeds.

**E03 minus E02:**

| Seed | Fisher RMSE delta | Separation RMSE delta (mas) |
| ---: | ---: | ---: |
| 11 | +1.332 | +0.527 |
| 23 | +2.572 | −0.052 |
| 47 | +0.744 | +0.554 |

**E03 is worse than E02 on global Fisher RMSE in all three matched seeds.** The
staged curriculum did not help. Its separation RMSE is also worse in two of three
seeds. But see the exposure confound below before attributing this to ordering.

### Family-stratified results — the aggregate is not a 50/50 average

The frozen validation artifact accepted **856 `joint_full_v4`** and **168
`radial_capture_v4`** ordered records — 83.6% / 16.4%, identical across all nine
runs — even though it was generated under the nominally 50/50 E02 policy. The
configured sampling weights are draw probabilities, and acceptance under
`require_same_dataset_family`, the bin weights and `max_sampling_attempts` does not
preserve them. **Every global S09 number is therefore dominated by
`joint_full_v4`.**

Mean over three seeds per experiment:

| Dataset family | Records | Metric | E01 | E02 | E03 |
| --- | ---: | --- | ---: | ---: | ---: |
| `joint_full_v4` | 856 | Fisher RMSE | 119.201 | **117.585** | 119.483 |
| | | MSE skill | 0.9022 | **0.9048** | 0.9017 |
| | | Separation RMSE (mas) | 4.267 | **3.828** | 4.194 |
| `radial_capture_v4` | 168 | Fisher RMSE | 102.557 | 99.343 | **98.782** |
| | | MSE skill | 0.8698 | 0.8778 | **0.8791** |
| | | Separation RMSE (mas) | **2.599** | 2.503 | 2.665 |

Two things are visible only in the stratified view:

- **E02's advantage over E01 is carried by the majority `joint_full_v4` family**
  (−1.62 RMSE, −0.44 mas), where E03 also loses. Adding radial training pairs
  improved performance on the *joint* population, not only on the radial one.
- **E03 is the best experiment on the `radial_capture_v4` family** (98.78 RMSE,
  skill 0.8791) while being the worst overall. Its curriculum did shift capability
  toward the radial population; that gain is simply outweighed by the 83.6%
  joint-family majority in the aggregate metric.

The radial family also has *lower* absolute Fisher RMSE but *lower* skill than the
joint family, because its mean correction magnitude is smaller — skill is the
baseline-relative quantity and is the fairer cross-family comparison.

### Distance dependence

E02 means by initial Fisher-distance bin (bin populations identical across all nine
runs):

| Initial Fisher distance | Ordered records | Independent pairs | MSE skill | Model RMSE | Baseline RMSE | Median rel. residual | Separation RMSE (mas) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–100 | **4** | **~2** | 0.754 | 8.39 | 17.15 | 0.478 | 0.327 |
| 100–250 | **16** | **~8** | 0.836 | 16.37 | 40.45 | 0.343 | 0.496 |
| 250–500 | 84 | ~42 | 0.922 | 24.33 | 87.36 | 0.265 | 0.925 |
| 500–1,000 | 238 | ~119 | 0.902 | 51.84 | 165.84 | 0.279 | 1.555 |
| 1,000–2,000 | 394 | ~197 | 0.897 | 106.28 | 331.89 | 0.290 | 3.242 |
| 2,000–5,000 | 288 | ~144 | 0.904 | 170.27 | 550.63 | 0.290 | 5.527 |

Only `radial_capture_v4` records populate the two smallest bins (4 and 16), and
because every record is reversal-coupled these are roughly **two and eight
independent samples**. No conclusion about the model's behaviour on very small
corrections is supportable in either direction. The apparent skill *decline* at
small distance (0.754, 0.836 against 0.90+ elsewhere) is exactly where the evidence
is weakest and should not be reported as a finding.

Across the well-populated bins skill is roughly flat at 0.90–0.92 while absolute
residual RMSE grows with displacement — the correction is proportionally about as
good at 3,000 as at 500, but the leftover error is far larger in absolute terms.

**Where E02 beats E01 (delta, negative improves):**

| Bin | Fisher RMSE delta | Separation RMSE delta (mas) |
| --- | ---: | ---: |
| 250–500 | −0.314 | +0.123 |
| 500–1,000 | +0.111 | +0.137 |
| 1,000–2,000 | −1.103 | +0.132 |
| **2,000–5,000** | **−3.468** | **−1.026** |

E02's advantage is concentrated in the **largest-displacement bin**, on both
metrics simultaneously, and it is there that the aggregate separation improvement
originates. That is consistent with the study's capture-range hypothesis: extra
large-displacement training pairs help most where the requested correction is
largest. In the mid-range bins the separation delta is slightly positive, so E02 is
not uniformly better — it trades a little mid-range accuracy for a substantial
large-displacement gain.

### Corrected Fisher-space and physical parameter metrics

Shares of residual squared error in the saved scaled-z coordinates, mean over three
seeds (sample SD in parentheses):

| Block | E01 | **E02** | E03 |
| --- | ---: | ---: | ---: |
| M1 Zernike coordinates | 43.6% (0.3) | **43.3% (0.2)** | 43.5% (0.7) |
| M2 Zernike coordinates | 55.8% (0.3) | **56.2% (0.1)** | 56.0% (0.7) |
| Source parameters + plate scale | 0.53% (0.09) | **0.49% (0.06)** | 0.50% (0.07) |

**The residual error is ~99.5% optical.** The source parameters the science case
actually cares about are among the best-corrected coordinates. E02 per-coordinate
results, mean over three seeds:

| Parameter | Fisher MSE skill | Physical RMSE | Unit |
| --- | ---: | ---: | --- |
| Binary separation | **0.9972** | **3.644** | mas |
| Log flux total | 0.9996 | 0.00140 | log10 photons |
| Contrast | 0.9923 | 0.0265 | dimensionless |
| Plate scale | 0.9971 | 4.07e-5 | arcsec/pixel |
| M1 Zernikes (range over Z4–Z11) | 0.839 – 0.999 | 0.41 – 5.28 | nm |
| M2 Zernikes (range over Z4–Z11) | 0.302 – 0.991 | 0.44 – 9.81 | nm |

> **Superseded interpretation.** An earlier reading of these results reported
> approximately **34% source/global residual share** and approximately **0.30
> binary-separation Fisher skill**. Both came from reading the alphabetically
> sorted `metrics.json` keys positionally: the value 196.706 attributed to binary
> separation is the M2 Z11 coordinate. The correct figures are **0.49%** and
> **0.9972**. The qualitative conclusion inverts — binary separation is not a weak
> coordinate, it is one of the strongest.
>
> **Physical separation RMSE values were never affected.** They are read from the
> named `physical_per_parameter_rmse` mapping by key, not by position, so the
> 4.042 / 3.644 / 3.987 mas figures and the 9.84% E02 improvement stand as
> previously reported. Global Fisher RMSE, MSE skill, slice metrics and
> distance-bin metrics are likewise label-independent and unchanged.

### M1/M2 residual degeneracy

Within the optical block, the share of residual squared error lying in the eight
equal-weight M1-minus-M2 difference coordinates `(e_M1,Zi − e_M2,Zi)/sqrt(2)` of the
stored scaled coordinates:

| Experiment | Difference subspace | Sum subspace |
| --- | ---: | ---: |
| S09-E01 | 89.0% | 11.0% |
| **S09-E02** | **88.7%** | 11.3% |
| S09-E03 | 89.0% | 11.0% |

The model recovers the combined wavefront far better than it decides which mirror
carries it. This independently reproduces on S09's mixed-family V4 population the
same structure S07 found (88.8%) on `joint_full_v4` alone and S06 found on V3, and
it is essentially unchanged by the radial mixture — **the mirror-assignment
degeneracy is not a capture-range problem and radial sampling does not touch it.**

**Caveat carried from S06.** This equal-weight scaled-coordinate difference
subspace is **not** the Schur-marginalized physical Fisher eigenbasis. S06's weak
physical eigenvectors are asymmetric — representative loadings near
(+0.472 M1, −0.881 M2) for Z4–Z6 and (+0.365 M1, −0.931 M2) for Z7–Z10 — and mix
several Zernikes, with mean squared projection onto the equal-weight difference
span of about 82.5%. The scaled-z → physical-theta transform remains unvalidated
and is not used here. The agreement between S06, S07 and S09 is qualitative support
for a mirror-assignment identifiability reading, not a validated coordinate-space
projection. See the [S06 review](../s06/review.md).

### Curriculum exposure — E03's comparison is confounded

E03's deficit cannot be attributed to easy-to-hard *ordering*, because the staged
policy also changes *cumulative exposure*. Expected fraction of all configured
training-pair draws per radial bin over the 600-epoch schedule:

| Radial bin | E02 | E03 | E03 / E02 | First epoch active in E03 |
| --- | ---: | ---: | ---: | ---: |
| 0–100 | 0.0833 | 0.1014 | **1.22×** | 0 |
| 100–250 | 0.0833 | 0.1014 | **1.22×** | 0 |
| 250–500 | 0.0833 | 0.1014 | **1.22×** | 0 |
| 500–1,000 | 0.0833 | 0.0736 | 0.88× | 100 |
| 1,000–2,000 | 0.0833 | 0.0736 | 0.88× | 100 |
| **2,000–5,000** | 0.0833 | 0.0486 | **0.58×** | 250 |

E03 receives approximately **58% as much expected 2,000–5,000 exposure as E02**,
and about 22% more exposure to the three smallest bins. Since E02's advantage over
E01 is concentrated in precisely the 2,000–5,000 bin, E03's worse aggregate score is
at least as consistent with *less hard-pair training* as with a harmful ordering
effect. **S09 does not isolate the curriculum-ordering hypothesis.** The
family-stratified result reinforces this: E03 is the *best* experiment on the radial
family it over-sampled at small distance, and the worst on the joint-family majority.

These are expected values from configured sampler probabilities and
`pairs_per_epoch`; realized accepted draws are not recorded.

## Interpretation and limitations

### Supported by these measurements

1. **Raising the configured radial mixture from 25% to 50% improves the learned
   initializer, most clearly on binary separation.** 9.84% mean separation RMSE
   improvement with all three matched seeds improving; 1.58% mean global Fisher RMSE
   improvement on two of three seeds.
2. **The benefit is concentrated at large displacement.** E02's largest gains over
   E01 are in the 2,000–5,000 Fisher-distance bin on both Fisher RMSE (−3.47) and
   separation RMSE (−1.03 mas), consistent with the capture-range hypothesis.
3. **The benefit transfers to the joint-state population**, not just the radial one:
   E02 beats E01 on `joint_full_v4` by 1.62 RMSE and 0.44 mas.
4. **The staged distance curriculum as configured did not improve aggregate
   performance** — worse than E02 on global Fisher RMSE in all three matched seeds —
   but the comparison is confounded with reduced hard-pair exposure and so does not
   refute easy-to-hard ordering as such.
5. **Residual error remains ~99.5% optical and ~88.7% mirror-difference-structured**,
   essentially unchanged across all three policies. Radial sampling does not address
   the mirror-assignment degeneracy.
6. **Source-parameter correction is strong in Fisher-scaled terms**: separation skill
   0.997, and all 1,024 ordered records improved over zero correction in every run.

### Not demonstrated

- **Behaviour on very small corrections.** The 0–100 and 100–250 bins hold 4 and 16
  reversal-coupled records (~2 and ~8 independent pairs). The study is effectively
  blind below 250.
- **Realized training mixtures.** Only configured sampling probabilities are
  recorded; accepted per-family and per-bin counts are not.
- **A clean curriculum-ordering result.** Ordering is confounded with ~58% relative
  exposure to the hardest bin.
- **Convergence.** All nine best checkpoints fall at epochs 573–599 of a 600-epoch
  budget.
- **Generalization to unseen registration nuisance.** The only slice is
  `radial_validation_seen_nuisance`.
- **Robustness to observation noise or forward-model mismatch.** Noise was disabled
  in every run and the forward model is the training model.
- **Microarcsecond astrometric retrieval.** A 3.6 mas separation residual is an
  initializer-quality figure for a single learned correction step — three orders of
  magnitude from µas astrometry. Do not conflate the two.
- **An ADORA capture radius or end-to-end convergence improvement.** Single-step
  Fisher-distance reduction on noiseless synthetic renders is not a capture-range
  result. The study's own distance thresholds are generic diagnostics.
- **Any locked-test performance.** `test_evaluated = false` in all nine manifests;
  no test evaluation was performed in this review.
- **Equivalence of the scaled-coordinate mirror-difference subspace with the true
  physical Fisher eigenvectors**, which are asymmetric and cross-coupled.

Absolute RMSE values here are not comparable with S07's (different validation
population and baseline) or S06's (V3).

## Reproducibility

All tables, figures and headline numbers above regenerate from the imported
artifacts by re-executing the local notebook:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 \
  Results/ml_analysis/S09/s09_capture_range_analysis.ipynb
```

The notebook is a thin driver over the maintained shared helpers in
[`campaign_analysis.py`](../campaign_analysis.py) and
[`campaign_plots.py`](../campaign_plots.py) (`load_campaign`,
`experiment_summary_table`, `matched_seed_delta_table`, `effect_size_summary_table`,
`distance_bin_delta_table`, `dataset_family_summary_table`,
`residual_error_budget_table`, `validation_cutoff_table`). Only genuinely
study-specific diagnostics — curriculum exposure schedules, the mirror-difference
subspace, reversal-coupling accounting and accepted-family composition — are
computed in it, directly from the saved scaled-z arrays. It reads
`Results/hpc_imports/ml/ls6/S09/`, writes to `Results/ml_analysis/S09/`, verifies the
archive manifest, asserts nine runs and `test_evaluated == false`, asserts the
recovered parameter order against the authoritative V4 contract, and performs no
inference, no test evaluation and no GPU work.

Generated locally (all paths under `Results/ml_analysis/S09/`, git-ignored):
21 CSV tables under `tables/` and 10 figures under `figures/`, plus
[`SUPERSEDED.md`](../../../../../Results/ml_analysis/S09/SUPERSEDED.md) recording which
outputs were regenerated and what changed.

## Follow-up and tracking

**Open identity questions.** The dirty working tree in all nine runs leaves the
executed code not fully pinned. Realized accepted training-pair proportions are
unrecorded; the sampler emits none. The scaled-z → physical-theta transform remains
unvalidated, so no residual projection into the S06 Schur eigenbasis is supported.

**Candidate follow-up directions** — these are research questions, not approved
campaigns. S10's eigenmode-weighted training results should inform which, if any,
is worth launching:

1. **Stronger radial sampling.** E02 beats E01 at 50/50 with no sign of saturation;
   75/25 and 100/0 radial ratios would locate the optimum.
2. **Exposure-equalized curricula at matched compute.** Rerun the staged policy with
   cumulative per-bin exposure matched to E02 (longer late stages, or a
   compute-matched budget) to separate ordering from exposure.
3. **Enriched small-distance validation.** Add independent, **non-reversed**
   small-displacement samples to a future evaluation artifact. The current 4 and 16
   reversal-coupled records make the <250 regime unmeasurable without touching the
   existing frozen artifact.
4. **Radial capture combined with targeted Schur weak-mode sampling.** Radial
   sampling left the 88.7% mirror-difference structure untouched; the two
   interventions address different failure modes and may compose.
5. **Downstream differentiable-optimization impact.** Measure whether the learned
   initializer actually improves ADORA convergence, rather than inferring capture
   from single-step Fisher-distance reduction.
6. **Scientific-priority metrics independent of aggregate Fisher RMSE.** E02 and E03
   rank differently on global Fisher RMSE, separation RMSE and per-family skill. A
   weighted objective reflecting science priority would make model selection less
   dependent on a metric dominated by optical nuisance modes and by the incidental
   83.6% joint-family composition of the validation artifact.

Do not tune against this frozen validation artifact. The locked test set remains
unevaluated and should stay that way until model selection is final.

**Tracking.** [ML study index and workflow](../README.md) ·
[campaign tracker notebook](../ml_campaign_tracker.ipynb) ·
[S07 review](../s07/review.md) (V4 joint-state reference) ·
[S06 review](../s06/review.md) (Schur/eigenmode reference) ·
[study prescription](../../s09/README.md).
