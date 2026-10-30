# S06 — V3 architecture/training bridge: scientific review

**Review date:** 2026-10-30; revised with Schur-marginalized eigenmode reference  
**Status:** First complete analysis of saved validation artifacts; ready for scientific review, not yet a final held-out test claim.

## Motivation and hypotheses

S01 established learnability of pairwise Fisher-scaled science-state corrections and a substantial benefit from longer training and a fixed learning rate of `1e-3`. S05 identified a larger `concat_diff` CNN as a provisional architecture improvement under a shorter training prescription. S06 combined those findings under the *same frozen V3 validation contract* to test reproducibility over matched seeds. Although the broad study name mentioned Fisher/eigenmodes, this three-experiment matrix **does not train in a Fisher eigenbasis**.

The specific comparisons are:

1. **Capacity:** E01 original CNN (~767,220 parameters) versus E02 larger CNN (~3,055,060 parameters), both at LR `1e-3`.
2. **Learning rate:** E02 versus E03, the same larger CNN at LR `2e-3`.
3. **Training duration:** Both architectures have up to 500 epochs (16,384 *ordered* training pairs per epoch, AdamW, same-nuisance/different-science pairs, reverse-pair augmentation). Three seeds: 11, 23, 47.

The frozen V3 validation artifact is `S01-VALIDATION-PAIRS-v1`, content SHA256 `68ccd41a35d286c8b060f291eef6c788a6b0d97c9660868f74e01b2b4feae499`, containing 2,048 ordered pairs split equally between seen- and held-out-nuisance populations; *nuisance is fixed within each pair*. Dataset `PREP-V3-v1`, prepared hash `4cdc325fbf8d4a0e07195ab075bea6f5035dfc01c9990cac03ee1f59c131e5e6`. All runs use the same true correction vectors and pair ordering. No test sets were evaluated.

## Main validation results

| Experiment | Seed 11 | Seed 23 | Seed 47 | Mean ± sample SD | Mean skill (MSE vs zero correction) |
| --- | ---: | ---: | ---: | ---: | ---: |
| E01 — original CNN, LR 1e-3 | 48.561 | 49.476 | 50.471 | 49.503 ± 0.955 | 0.9610 |
| **E02 — larger CNN, LR 1e-3** | **47.323** | **48.369** | **48.577** | **48.090 ± 0.672** | **0.9632** |
| E03 — larger CNN, LR 2e-3 | 48.627 | 48.798 | 49.246 | 48.890 ± 0.320 | 0.9620 |

The frozen V3 zero-correction Fisher RMSE is approximately 250.84.

The improvement of E02 over E01 is 1.24, 1.11, and 1.89 Fisher-RMSE units for matched seeds; the reduction in the means is 2.85%. E02 also outperforms E03 for all three seeds, suggesting no benefit to doubling LR under this prescription. These are modest differences at three seeds, not proof of broad optimizer/model superiority.

Eight runs reached epoch 499 (500 total epochs). E01-R003 stopped legitimately at epoch 434 (435 total epochs) due to early stopping; its best checkpoint was at epoch 354. E02 best epochs: 472, 498, 477; most best checkpoints occur late. Thus the 500-epoch schedule should not be described as converged.

## Correction behavior and evaluation distribution

For E02, mean MSE skill is ~0.9632, with ~97.3% of validation pairs receiving a correction closer to the target in Fisher-space Euclidean distance (`||error|| < ||true correction||`). This is a **single-step synthetic validation** result; it is not a demonstrated ADORA initialization capture range.

| Initial Fisher distance | Validation pairs | E02 MSE skill | E02 fraction improved |
| --- | ---: | ---: | ---: |
| 0–100 | 6 | 0.968 | 1.000 |
| 100–250 | 168 | 0.794 | 0.923 |
| 250–500 | 360 | 0.839 | 0.960 |
| 500–1,000 | 702 | 0.890 | 0.961 |
| 1,000–2,000 | 716 | 0.972 | 1.000 |
| 2,000–5,000 | 96 | 0.993 | 1.000 |

Bin-level statistics average matched three-seed values. Small-distance and large-distance comparisons should account for very uneven bin counts, differences in the perturbation population, and the fact that fractional skill can improve with larger initial displacement even if residual absolute error does not.

E02 validation RMSE is 47.59 ± 0.32 on heldout-science/heldout-nuisance pairs and 48.58 ± 1.02 on heldout-science/seen-nuisance pairs. These two subsets contain different pairs; the small gap is not evidence of invariance to *changes* in nuisance, because the S06 pairing policy holds nuisance fixed within each image pair.

## Where the errors remain

For E02, the *Fisher-scaled coordinate* squared-error budget averages ~0.8% in source parameters plus plate scale, ~48.0% in M1 Zernike coefficients, and ~51.2% in M2 coefficients. These percentages describe squared prediction errors in the stored 20-dimensional target coordinate system, not physical WFE (nm), not formal estimation uncertainty, and not the eigenmode spectrum of an optical Fisher matrix.

The error concentration is not evenly distributed across optical coefficients. The E02 three-seed mean residual RMSE is large for mirrored coefficient indices 0, 1, 2, 5, and 6. M2 has notably lower per-coordinate MSE skill for those indices (~0.71, 0.77, 0.73, 0.75, 0.86) than M1 (~0.93, 0.96, 0.94, 0.94, 0.97), although the absolute residual RMSEs are often comparable.

### Mirror-pair residual covariance

For coefficient index `i`, define the residual in stored scaled science coordinates as `e_M1,i` and `e_M2,i`. This is the network prediction minus the prescribed science correction (in z coordinates). Across the three E02 seeds, the mean residual correlations are:

| Matched index | Corr(e_M1,i, e_M2,i) |
| --- | ---: |
| 0 | −0.857 |
| 1 | −0.967 |
| 2 | −0.975 |
| 3 | −0.145 |
| 4 | −0.007 |
| 5 | −0.971 |
| 6 | −0.970 |
| 7 | −0.719 |

The corresponding *true paired parameter corrections* have cross-mirror correlations close to zero, so the strong residual anti-correlation is not simply inherited from paired target sampling.

Define the equal-weight combinations within the **stored Fisher-scaled z representation**:

    e_plus,i  = (e_M1,i + e_M2,i) / sqrt(2)
    e_minus,i = (e_M1,i - e_M2,i) / sqrt(2)

Across E02 seeds, 95.5%, 95.9%, and 96.6% of the *optical* residual squared error is in the collection of `e_minus` coordinates. This is unusually compelling evidence for an assignment ambiguity between two mirrors in the present modeling and training setup. However, the equal-weight difference directions are not necessarily optical Fisher eigenvectors, especially when physical coefficients have different scales or the full information matrix mixes mirror and source parameters.

A PCA of the centered 16-dimensional optical residual vector finds that the first five residual PCs account for ~93.1% of the optical residual covariance. Their leading loadings are mostly anti-aligned matched M1/M2 coefficients. These are **residual covariance PCs**, not information eigenmodes. The **direct quantitative overlap of these scaled-z residual PCs with physical Schur eigenspaces** remains to be measured; the independent physical FIM cross-check below provides qualitative support but not a validated coordinate-space projection.


## Nuisance-marginalized Fisher eigenmode cross-check (2026-10-30)

### Provenance and comparison boundary

The companion eigenmode products are a **V3 flight-simple, 50-ms, PSF160**
optical sensitivity analysis, originally under
`Results/ml_eigenmodes/ml_dataset_v3_flight_simple_50ms_psf160/`
(`sweep_schur_marginalized`). These files provide `fim_matrix.csv`,
`eigenvalues.csv`, `eigenvectors_long.csv`, `parameter_labels.csv`, and
`manifest.json`. This is supporting model-level evidence, not proof that its
Fisher matrix is the exact evaluation metric behind every S06 validation pair;
confirm the optical and basis identity before any numerical joint projection.

The matrix is the **20×20 Schur-reduced Fisher information** for the science and
optical parameters, after marginalizing three registration nuisance parameters
(position angle, binary X, binary Y). The original computation used a
pseudoinverse with `rcond=1e-12` and reported no negative eigenvalues.
The eigenvectors are unit-Euclidean in the **physical packed basis**, with
unlike units (arcseconds, log counts, dimensionless contrast, arcseconds/pixel,
and nanometers), not whitened or normalized per parameter. Because of those
units, do not interpret raw global eigenvalue ratios as a unit-invariant
ranking of scientific importance. Comparisons among WFE-only directions are
more interpretable because M1/M2 coefficients are all in nanometers.

S06's prediction residuals, by contrast, are stored in **Fisher-scaled `z`
coordinates**. Their approximately 96% energy in equal-weight M1–M2
*difference coordinates* does **not** mean 96% in the weak physical Fisher
eigenspace. A validated mapping between `z` and physical `theta` is required
before calculating that overlap.

### What the Schur spectrum and eigenvectors show

The 20 eigenvalues separate into four large source/plate-scale dominated modes,
eight stronger optical modes (eigenvalues **890.7–2854.7** in the exported
physical basis), and **eight very weak optical modes** with eigenvalues
**0.003871–0.027846**. The separation between the smallest stronger-optical
eigenvalue and the largest weak-optical eigenvalue is approximately **32,000×**.
The manifest's eight `near_zero` flags use an explicit numerical threshold
(`1e-12 × global largest eigenvalue ≈ 0.2398`); this is **not** evidence of
an exact rank-8 null space. All exported eigenvalues are positive.

For the matched pairs Z4–Z10, the Schur-FIM normalized cross sensitivity

    C_i = F[M1 Zi, M2 Zi] / sqrt(F[M1 Zi, M1 Zi] F[M2 Zi, M2 Zi])

is **0.99991–0.99997**. In a weighted-Jacobian interpretation these two
parameters produce almost indistinguishable local image perturbations after
registration nuisance is marginalized. This independently substantiates the
mirror-assignment ambiguity suggested by S06 prediction errors.

However, the weak physical eigenvectors are **not simply**
`(M1 Zi − M2 Zi)/sqrt(2)`. The per-pair near-null directions for Z4–Z6
have representative normalized coefficients around **(+0.472 M1,
−0.881 M2)**, while Z7–Z10 are closer to **(+0.365 M1, −0.931 M2)**.
This asymmetry reflects differing Fisher sensitivity magnitudes on the two
mirrors. Moreover, the actual 20D eigenvectors mix several Zernikes. Across
the **eight weak eigenvectors**, their mean squared projection onto the span
of eight *equal-weight* M1-minus-M2 directions is about **82.5%** (individual
modes range **47.8–90.3%**). That is strong subspace alignment, not equality.

The notable exception to one-to-one matching is **Z11**. Its same-index
M1/M2 Schur cross-correlation is only **+0.280**, while M1 Z4 and M2 Z11
have strong **negative** cross sensitivity (approximately **−0.960**).
The weakest mode (rank 20, eigenvalue ~0.00387) is dominated by **M2 Z4
(−0.769)** and **M2 Z11 (−0.594)**, with smaller M1 contributions.
This illustrates why storing and reviewing the full eigenvectors is more
informative than inspecting matched-mirror pairs alone.

### Relationship to the learned S06 errors

The S06-E02 prediction-error correlations for matched Z4–Z11 are
`−0.857, −0.967, −0.975, −0.145, −0.007, −0.971, −0.970, −0.719`.
Z4–Z6 and Z9–Z10 strongly show the mirror trade-off anticipated from the
FIM, but Z7/Z8 do not have a comparably strong residual anticorrelation
**despite** their nearly collinear local sensitivities. The two metrics
measure different objects: Fisher sensitivity geometry at an optical
reference versus a trained model's error covariance over a finite
validation-pair distribution. They should not be equated.

This strengthens the interpretation that S06's remaining difficulty is a
structured optical identifiability problem, while qualifying the earlier
shorthand that the weak modes are isolated equal-weight M1/M2 differences.

### Practical implications for future campaigns

1. **Freeze this Schur product as a reference**, retaining its original
   physical-basis normalization, parameter order, provenance, and content hashes.
   The original eigenmode sweep remains authoritative; this S06 directory has
   a clearly labeled analysis copy.
2. Before any learned weak-mode projection, **establish the exact linear
   transform** from the S06 stored `z` coordinates into physical `theta` and
   check reference-state, dimension, and basis identity. Also confirm that
   the eigenmode product uses an optical configuration comparable to S06.
3. For training-data enrichment, **sample along actual weak physical
   combinations**, not only individual Zernikes or equal-amplitude mirror
   differences, then construct images using physically consistent targets.
   Include paired mirror perturbations with explicit strength and distance
   controls and keep the original validation data untouched.
4. An exact physical degeneracy cannot be removed by oversampling identical
   measurements. Test what optical diversity (field angle, wavelength,
   registration, focus, observation geometry, prior information) raises the
   information content along weak directions before interpreting improved
   neural regression as evidence of new identifiability.
5. Keep the S10 eigenmode-weighted-loss experiment conceptually separate:
   its artifact and optical normalization must be independently checked
   before treating S10 as a direct numerical comparison with this V3 product.

**Companion derived files (local, under `Results/ml_analysis/S06/`):** `tables/schur_eigenmode_summary.csv`,
`tables/schur_weak_eigenvectors_physical_basis.csv`,
`tables/matched_mirror_fim_vs_s06_residuals.csv`, and three new figures.
Analysis copies (including the full eigenvectors) are stored locally under
`source_eigenmodes/` with SHA256 hashes for provenance.

## Interpretation, cautions, and next experiment gate

1. **S06 established a reliable V3 bridge baseline:** the larger network at LR `1e-3` wins on best-checkpoint validation RMSE across all three matched seeds, but the capacity gain is small relative to a fourfold parameter count.
2. **Training duration mattered:** most models still find new best checkpoints late, and a 100–300 epoch comparison would not reflect the eventual outcome.
3. **The remaining error is strongly structured:** the decisive new feature is matched-mirror *anti-correlation*, not merely large M2 residual RMSE.
4. **Data enrichment alone may not resolve an exact degeneracy:** extra pairs along a weak direction provide a clearer loss signal only insofar as images actually distinguish changes along that direction. If the optical forward model maps a difference direction into a near-null image perturbation, field variation, time diversity, alternate wavelength/pupil state, more informative target combinations, physical priors, or explicit uncertainty modeling may be needed.
5. **Do not tune to the frozen validation set:** derive target directions from independent optical-model information and sample new training pairs; preserve the old validation contract and a separate untouched test set for final selection. Analyze performance within each fixed-distance and nuisance slice, with strong-science-mode degradation reported explicitly.
6. **S10 already tests broader strong/weak eigenmode loss weighting on V4.** Complete S10 first, then consider a controlled factorial of targeted pair sampling and bounded eigenmode-weighted loss at equal training compute, with fixed paired seeds.

Suggested diagnostics before new training: validate the optical-configuration match and the linear coordinate map from S06 scaled `z` residuals to the supplied Schur-FIM physical parameter basis; then compute actual eigenspace residual projections and MSE skill, and test whether additional image diversity increases information along the weakest directions. The independent Schur reference establishes the local sensitivity pattern, but does not replace these sample- and basis-matched tests.

## Reproducibility and file references

- Input: `s06_s09_review_20261008.tar.gz` (uploaded compact LS6 results archive; source data under `runs/S06/<experiment>/<run>/`).
- Analyzed files per run: `metrics.json`, `history.csv`, `run_manifest.json`, `run_config_resolved.json`, and `evaluation_predictions.npz`.
- Supporting calculated CSVs: `runs.csv`, `parameters.csv`, `slices.csv`, `distance_bins.csv`, `matched_zernike_cross_correlations.csv`; figures `s06_validation_rmse_by_seed.png`, `s06_e02_parameter_rmse.png`, `s06_skill_vs_distance.png`, `s06_e02_optical_residual_correlations.png`.
- Repository SHA recorded by all S06 run manifests: `5fa65027f1fe4c7495f9be292303d25fce82fe4e`, `git.dirty = true`. Exact working-tree diff not in the compact archive.
- Run manifests `test_evaluated = false`. Claims are from frozen **validation** only and need final held-out test evaluation on locked experiments.
- This report uses the saved parameter order `[separation, log_flux, contrast, plate_scale, M1 coeff 0–7, M2 coeff 0–7]`, independently verified against per-parameter RMSE values in each run's `metrics.json`.


## Integrated support and weak-mode table

See [support and provenance](support.md) for checksums, reproducible local paths,
analysis-copy status, and the eight weak-mode loadings. This is the authoritative
S06 scientific review; the dated import package is noncanonical input.

**Provenance reconciliation:** the original local work mirror was an earlier
experiment generation, with a different source revision and shorter budgets.
It is preserved as historical data outside active discovery. The production
archive was recovered locally and its nine saved validation predictions
reproduce this review; those artifacts are now the active S06 import. See
[the reconciliation](reconciliation.md) for hashes, identities and limitations.
The dirty-worktree provenance warning remains applicable.
