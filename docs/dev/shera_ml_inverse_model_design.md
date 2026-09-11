# SHERA ML Inverse-Model Design and Experiment Roadmap

**Status:** Active ML experiment ledger and roadmap
**Date:** 2026-09-08
**Scope:** ML-assisted state estimation and initialization for the SHERA/ADORA differentiable optical model

## 1. Purpose

This document captures the design direction for machine-learning experiments built around the SHERA forward model. The goal is not to replace the differentiable physics model or ADORA with a black-box estimator. The primary hypothesis is that a learned image-based update operator may extend the capture range of the existing physics-based inference, especially in nonlinear or poorly initialized regimes such as early on-orbit commissioning.

The central proposed task is therefore:

\[
\left[I_{\mathrm{model}}, I_{\mathrm{obs}}\right]
\longrightarrow
\widehat{\Delta\theta},
\]

where the model estimates a correction from the current model state toward the observed state. The learned correction can then be passed to ADORA as an improved initialization rather than treated as the final scientific estimate.

The roadmap deliberately keeps the first experiments simple and interpretable. More elaborate contrastive, JEPA-like, uncertainty-aware, or iterative learned-optimizer ideas should be added only when simpler baselines establish a need.

## 2. Design principles

1. **Preserve the physics model as the authoritative inference model.** ML should initially serve as an initializer, nonlinear correction, or representation-learning aid.
2. **Benchmark against the linear/Fisher estimator.** A learned model is only useful if it adds value beyond the local estimator already implied by the differentiable forward model.
3. **Use physical parameter outputs by default.** Keep the final regression target interpretable in the physical state basis, with explicit normalization.
4. **Use Fisher/eigenmode information as structure, not decoration.** The FIM should inform scaling, diagnostics, dataset design, loss weighting, and hard-direction analysis.
5. **Treat nuisance robustness as a first-class problem.** Translation/rotation nuisance variables should be tested through both implicit invariance and explicit multitask estimation.
6. **Separate canonical data from ML working data.** Original high-precision FITS and metadata remain authoritative; optimized training representations are reproducible derivatives.
7. **Avoid a model zoo.** Follow a small number of self-contained notebook experiments, each changing one important assumption at a time.
8. **Use grouped validation/test splits.** Prevent leakage across repeated nuisance realizations, repeated physical states, or related generated samples.
9. **Track experiment provenance from the start.** Dataset version, split policy, model configuration, loss, seed, normalization, and metrics should be recoverable for every result.

## 3. Existing data and current baseline work

The existing V3 training-dataset workflow is already a strong foundation. It is plan-first, uses Fisher-diagonal parameter scales, writes self-describing manifests/sample metadata, and supports pair-grid, nuisance-replicate, and sparse-mixture concepts. The current pair-grid datasets are therefore useful both as scientific sensitivity atlases and as ML training/evaluation data.

Wave 1 reusable dataset infrastructure is documented in
`docs/dev/ml_prepared_dataset_wave1.md`.  It implements derived `.npy` array
shards, sample-centric JSONL indexes, vector-space metadata, coordinate
transforms, fidelity validation, and deterministic grouped splitting while
leaving model-specific pair sampling and training code to later experiment
layers.

The previous analysis used the pair-grid data primarily as a **20-way multi-label classification problem**: identify which two parameters were controlled in a rendered image. They contain several reusable ideas:

- compact CNN encoders;
- spatial-attention pooling experiments;
- normalized latent embeddings;
- handcrafted physics-inspired baselines;
- supervised contrastive losses for selected M1/M2 degeneracies;
- memory-conscious FITS loading and out-of-core processing.

The current classification formulation is not the intended long-term target. It ignores perturbation amplitude, can assign physically ambiguous labels at zero-amplitude grid points, and does not directly match the intended ADORA initialization task. The new work should reuse the useful implementation patterns while changing the learning objective to continuous state correction.

As of 2026-09-08, the S01 and S05 V3-benchmark development runs have completed
on TACC Lonestar6, and the V4 raw corpus has completed production rendering on
Gattaca2. The program has therefore moved from infrastructure validation,
learnability, first optimizer study, and first architecture study into
large-corpus preparation, nuisance-robust training design, capture-range
curriculum, and scalable model development. The V3 benchmark remains valuable
as a frozen regression/comparability benchmark; V4 is the next main training
corpus after its prepared-dataset layer is implemented.

## 4. Primary ML task: Siamese state-correction regression

### 4.1 Input formulation

Use a shared-weight Siamese encoder:

\[
h_A = E(I_A), \qquad h_B = E(I_B),
\]

where the same encoder processes the model and observed images.

A simple first comparison head can consume

\[
[h_A,\,h_B,\,h_B-h_A]
\]

and predict the correction between the two physical states.

The two inputs are interpreted operationally as:

- \(I_A = I(\theta_{\mathrm{model}})\): image rendered from the current model state;
- \(I_B = I_{\mathrm{obs}}\): observed image or a simulated truth image during training.

The model predicts

\[
\Delta\theta = \theta_B-\theta_A.
\]

This formulation avoids dependence on a single nominal reference image. Any two compatible dataset states can form a supervised pair.

### 4.2 Reverse mapping

Every ordered pair provides a free reversed example:

\[
(A,B)\rightarrow\Delta\theta,
\]

\[
(B,A)\rightarrow-\Delta\theta.
\]

The reverse pair should be included either through data augmentation or explicit paired batching. An optional consistency loss can later enforce

\[
f(A,B)+f(B,A)\approx0.
\]

This is an antisymmetry/equivariance constraint, not itself a conventional contrastive-learning objective.

### 4.3 Identity mapping

Same-image or same-state pairs provide a second useful physical constraint:

\[
f(A,A)\approx0.
\]

This can be included naturally in batches containing same-state nuisance/noise variants.

### 4.4 S01 production workflow

The S01 production prescription is tracked under `work/experiments/ml/s01/`.
The generic ML package and scripts remain reusable; the S01 work directory
resolves a compact `study.yaml` into the ordinary
`train_pairwise_correction(...)` config.

As of 2026-09-08, the three-seed `S01-E01` baseline has completed on TACC
Lonestar6: `S01-E01-R001` seed 11, `S01-E01-R002` seed 23, and
`S01-E01-R003` seed 47 completed under LS6 jobs 3418678, 3418707, and 3418708.
The zero-correction Fisher RMSE was 250.840; best validation RMSEs were
72.4681, 75.7559, and 74.3278, giving mean 74.1839 and sample standard
deviation 1.6486. These are validation results because ordinary S01
development used `evaluate_test: false`.

The later seed-11 optimizer/training-control wave `S01-E02` through `S01-E07`
also completed. The best current optimizer/training-control candidate is
`S01-E03`: fixed LR 1e-3, 300-epoch maximum, best epoch 273, best validation
RMSE 57.8676, and MSE skill 0.946780. Relative to `S01-E01-R001`, this reduces
best validation RMSE by approximately 20.1%. It is not a convergence proof:
`S01-E03`, `S01-E02`, `S01-E06`, and `S01-E07` reached their best checkpoints
late in the allowed training interval.

For `S01-E01-R001`, validation and test pairs are deterministic frozen
artifacts.  Validation is used for checkpoint selection, early stopping, and
model-development decisions.  Test is materialized and identity-checked, but is
not evaluated during ordinary training (`evaluate_test: false`).

The canonical reusable S01 pair policy is `s01_clean_same_pair_grid_v1`: same
registration nuisance, different science state, same original V3 pair grid,
Fisher distance in `[0, 5000]`, and explicit reverse ordered-pair augmentation.
Batch size counts ordered pair examples, not individual images.

Image scaling uses one train-derived `global_max_abs` scalar applied to all A/B
images.  It preserves relative image amplitude and flux information and is not
per-image normalization.

S01 distance diagnostics are config-driven with default edges
`[0, 100, 250, 500, 1000, 2000, 5000]`.  Bins are left-closed/right-open except
for the final bin, which includes exactly the upper edge.  Samples outside the
configured range are reported separately.

Training checkpoints preserve early-stopping state.  `checkpoint_best.pt`
updates on any strict validation-loss improvement, while the patience counter
resets only on a configured meaningful relative improvement.

Lonestar6 is now validated as a second execution site alongside Gattaca2 for
this ML workflow.  The strict GPU preflight reproduced the canonical prepared,
split, validation-pair, and test-pair identities.  A one-epoch real-data smoke
test on an NVIDIA A100-PCIE-40GB completed successfully, but that smoke loss is
infrastructure evidence only and is not an S01 scientific result.

Historical infrastructure events remain separate from scientific results. LS6
job 3418670 failed immediately for `S01-E01-R001` before the accepted
scientific run, and `S05-E01-LS6-SMOKE` job 3419329 was cancelled before
execution.

## 5. Parameter representation and Fisher scaling

### 5.1 Physical output basis

Keep the network output in the physical/scalarized science-parameter basis. Let

\[
\Delta\theta\in\mathbb{R}^{P}
\]

contain the scalarized state correction.

Use the Fisher-based parameter scale already stored by the dataset generator:

\[
z_i = \frac{\Delta\theta_i}{\sigma_{\mathrm{FIM},i}}.
\]

The primary regression target is therefore

\[
\Delta z_i
=
\frac{\theta_{B,i}-\theta_{A,i}}
{\sigma_{\mathrm{FIM},i}}.
\]

This gives every output an approximately comparable local information scale while preserving a direct invertible mapping to native physical units.

### 5.2 Fixed reference scaling

For the first experiments, use one documented reference vector of \(\sigma_{\mathrm{FIM}}\) values associated with the dataset/reference model. Do not recompute a different scaling for every sample. Stable coordinates are more useful for training and interpretation.

The metadata should record:

- parameter order;
- native units;
- nominal/reference value;
- \(\sigma_{\mathrm{FIM}}\);
- FIM provenance/configuration;
- nuisance treatment used when the FIM was computed.

## 6. Eigenmode strategy

### 6.1 Three nuisance treatments

Previous SHERA eigenmode exports include three meaningful FIM constructions:

1. **Full FIM including nuisance variables.** Useful for understanding joint science-registration degeneracies and possibly for a multitask model.
2. **Fixed-nuisance science FIM.** Useful as an idealized diagnostic when registration is assumed known.
3. **Nuisance-marginalized Schur-complement FIM.** Likely the most relevant default science eigenbasis when the desired science representation should account for uncertainty/absorption by nuisance registration.

These are not contradictory eigensystems; they answer different questions. Every ML experiment using an eigenbasis must record which construction it uses.

For S10-v1, freeze the basis choice to the nominal 20-dimensional science FIM
in the same science-parameter ordering and Fisher-sigma convention used by
PREP-V4-v1. Registration nuisance parameters are held fixed. The physical
science FIM \(F_\theta\) is transformed before eigendecomposition with
\(F_z=D F_\theta D\), where \(D=\mathrm{diag}(\sigma_{\mathrm{FIM}})\) and
\(\Delta\theta=D\Delta z\). The training-consumed `ScienceEigenbasis` is
therefore a basis of the prepared Fisher-scaled science coordinate, not a
physical-theta basis. A nuisance-marginalized Schur-complement basis remains a
later comparison, not part of the S10-v1 launch campaign.

### 6.2 Diagonal Fisher scaling is not eigen-whitening

There are three distinct operations that should not be conflated.

#### A. Diagonal Fisher scaling

Define

\[
z=D^{-1}\Delta\theta,
\qquad
D=\mathrm{diag}(\sigma_{\mathrm{FIM},1},\ldots,\sigma_{\mathrm{FIM},P}).
\]

The FIM in these coordinates is

\[
F_z=D^T F_\theta D.
\]

If \(\sigma_i\approx1/\sqrt{(F_\theta)_{ii}}\), then the diagonal entries of \(F_z\) are approximately unity. This removes gross per-parameter scale differences but does **not** remove correlations.

#### B. Eigen-rotation

Diagonalize the scaled FIM:

\[
F_z=Q\Lambda Q^T.
\]

The unwhitened eigen coefficients are

\[
a=Q^Tz.
\]

This rotates into statistically meaningful coupled directions but leaves their different curvatures/eigenvalues intact.

#### C. Eigen-whitening

The existing ADORA eigenmode machinery can additionally scale eigen-directions by \(1/\sqrt{\lambda_k}\), yielding optimizer coordinates with approximately unit local curvature. In the equivalent coefficient view, a whitened coordinate scales the unwhitened eigen coefficient by \(\sqrt{\lambda_k}\).

Thus diagonal Fisher scaling **before** eigendecomposition and full eigen-whitening **after** eigendecomposition are not equivalent. The recommended ML diagnostic pipeline is:

1. Fisher-scale the physical parameters;
2. form the appropriately fixed/full/Schur science FIM in those scaled coordinates;
3. eigendecompose it;
4. retain both the unwhitened eigen coefficients and optional whitened coordinates for diagnostics.

### 6.3 Recommended output and loss strategy

Do **not** replace the main physical regression output with eigenmode labels initially. A complete orthogonal eigen-rotation with ordinary MSE is only a change of basis.

Instead, predict \(\hat z\) in physical/Fisher-scaled coordinates and project the error into the eigenbasis:

\[
e=\hat z-z,
\qquad
e_{\mathrm{eig}}=Q^Te.
\]

Report both physical-coordinate and eigen-coordinate errors.

An optional eigenmode-aware loss is

\[
\mathcal L_{\mathrm{eig}}
=
\frac{1}{P}
\sum_k w_k e_{\mathrm{eig},k}^2.
\]

Candidate weighting experiments include:

- \(w_k=1\): diagnostic equivalent to unweighted scaled-physical MSE for a complete orthogonal basis;
- \(w_k\propto\lambda_k^{\alpha}\): emphasize well-constrained/high-information directions and approximate a Fisher/image-space metric;
- \(w_k\propto(\lambda_k+\lambda_{\mathrm{floor}})^{-\alpha}\): deliberately emphasize weak directions so the model cannot minimize aggregate loss mainly by fitting easy modes.

Use a floor/cap for inverse-eigenvalue weighting to avoid letting nearly-null modes dominate numerically.

Important distinction: **plain MSE in fully whitened coordinates weights physical eigen-errors approximately by \(\lambda_k\)** and therefore emphasizes strong modes, not weak modes. Weak-mode emphasis should be an explicit loss design rather than assumed to follow automatically from whitening.

### 6.4 Eigenmodes as dataset diagnostics and design axes

Eigenmodes may be more valuable for dataset design than for output labels. Use them to:

- identify hard M1/M2 differential directions;
- stratify validation/test sets by eigenvalue or weak-mode content;
- create dedicated stress datasets along selected weak eigenvectors;
- quantify error as a function of eigenmode rank/eigenvalue;
- compare ML performance against local Fisher predictions;
- determine whether nonlinear image structure becomes informative along directions that are locally weak.

For a selected mode \(q_k\), render/simulate states along

\[
z=\alpha q_k
\]

for a range of amplitudes \(\alpha\). These focused sweeps can complement the space-filling V4 dataset.

## 7. Nuisance treatment and representation learning

Let the state be decomposed into science/slow parameters \(\theta\) and registration nuisance parameters

\[
\phi=[x,y,\mathrm{PA}].
\]

Test a deliberately small set of strategies.

### Experiment N1: implicit nuisance invariance

Randomize nuisance states but train only the science correction head:

\[
[I_A,I_B]\rightarrow\Delta z_{\mathrm{science}}.
\]

The encoder must learn whatever nuisance robustness is required by the task.

### Experiment N2: multitask nuisance estimation

Use a shared encoder with two output heads:

\[
[I_A,I_B]\rightarrow
[\Delta z_{\mathrm{science}},\Delta\phi].
\]

This tests whether explicitly representing registration improves science-state recovery.

### Experiment N3: multitask + science-embedding invariance

For samples with identical science state but different nuisance realization,

\[
I_1=I(\theta,\phi_1),
\qquad
I_2=I(\theta,\phi_2),
\]

encourage a designated science embedding to satisfy

\[
h_{\mathrm{sci}}(I_1)\approx h_{\mathrm{sci}}(I_2).
\]

Do not force the entire latent representation to be nuisance-invariant if a nuisance head must retain \(\phi\) information.

This is the main JEPA-/joint-embedding-inspired experiment: learn a representation that preserves physical instrument state while discarding nuisance/noise details irrelevant to the science correction.

### Noise invariance

Independent noise realizations of the same underlying physical+nuisance state provide especially clean positive pairs. Once physical noise is enabled, the science representation should generally be encouraged to ignore stochastic noise realization.

## 8. Existing V3 dataset as the first Siamese training set

The current nuisance-replicated pair-grid dataset can answer several architecture questions without rendering V4 first.

For each image, reconstruct a dense Fisher-scaled state vector. A pair-grid sample has up to two nonzero science coordinates; dynamically pairing two arbitrary states can therefore produce correction vectors with up to four nonzero coordinates.

Useful pair categories are:

1. **same science state, different nuisance**: \(\Delta\theta=0\); ideal invariance/identity examples;
2. **different science state, same nuisance**: isolates science correction;
3. **different science state, different nuisance**: realistic combined problem;
4. **reverse pair** for every non-identical pair.

Dynamic pair generation avoids materializing a combinatorial pair table.

## 9. Data preprocessing and memory/I/O infrastructure

### 9.1 Canonical vs working representation

The original generated FITS images and JSON/manifest metadata remain the **canonical dataset** and should not be modified or replaced by ML preprocessing.

Create a reproducible derived ML dataset:

```text
canonical V3/V4 FITS + metadata
            |
            v
      ML preprocessor
            |
            +-- index.parquet
            +-- schema.json
            +-- preprocessing_manifest.json
            +-- shards/
                +-- images_00000.npy
                +-- images_00001.npy
                +-- ...
```

### 9.2 Precision policy

The source FITS images are expected to preserve the original high-precision rendering (typically float64 in the current workflow). The default ML shard representation may be float32 for memory, I/O, accelerator compatibility, and training throughput, but this is explicitly a **lossy working representation**.

Policy:

- preserve canonical float64 FITS;
- default ML shards to float32 unless validation shows a meaningful degradation;
- record source dtype and shard dtype;
- quantify casting error during preprocessing;
- optionally write or retain a small float64 validation shard/subset;
- never use float16/bfloat16 as the archival representation.

The preprocessor should compute at least a dtype audit over a representative sample:

- max absolute pixel error;
- RMS pixel error;
- relative error with a sensible floor/mask;
- total-flux difference;
- optional centroid/low-order image-moment differences;
- fraction of pixels changed beyond configurable tolerances.

A small downstream A/B experiment should compare float64-read-at-runtime and float32-shard training/evaluation before declaring float32 permanently safe. In practice, most PyTorch/GPU training will operate in float32 or mixed precision anyway, so preprocessing to float32 is likely advantageous, but the decision should remain evidence-based.

### 9.3 Sharding

Avoid repeatedly opening hundreds of thousands of FITS files during every epoch.

Initial recommendation:

- uncompressed fixed-shape `.npy` shards;
- approximately 512--2048 images per shard, with 1024 as a reasonable first default;
- memory-map shards read-only;
- maintain a small LRU cache of open/mapped shards;
- store `shard_id` and `shard_offset` in the Parquet index.

Do not start with one monolithic array if the execution environment has address-space or memory-map constraints. Small shards bound virtual mappings and make partial dataset access easier.

Compression/Zarr/HDF5/WebDataset can be revisited only if the simple `.npy` solution proves insufficient.

### 9.4 Parquet index

The ML index should contain one row per underlying rendered image, not one row per Siamese pair.

Recommended columns include:

```text
sample_id
dataset_version
dataset_family
sample_role
source_fits_path
source_metadata_path
shard_id
shard_offset
source_dtype
shard_dtype
pair_id / grid indices when applicable
nuisance_id
science_state_native
science_delta_native
science_state_fim_scaled
science_delta_fim_scaled
nuisance_state_native
parameter_schema_id
fim_basis_id
eigenbasis_id if available
sampling provenance
noise provenance
group/split identifiers
```

Dense vectors may be stored as fixed-size/list columns or as flattened named columns; choose the representation that is easiest to inspect and robustly load.

### 9.5 Preprocessor validation

The preprocessor should support:

- dry-run/index-only mode;
- configurable shard dtype;
- configurable images per shard;
- resume/restart;
- sample-level source hashes or stable provenance identifiers;
- random round-trip validation comparing shard pixels to source FITS;
- shape and finite-value checks;
- a summary of total images, bytes, dtype, shard count, and conversion statistics.

### 9.6 V4 prepared-dataset layer

The next major data implementation task is a V4 prepared-dataset layer. It
should preserve the canonical raw FITS + JSON corpus as authoritative while
deriving efficient training arrays or shards. Reuse or generalize the existing
prepared-dataset infrastructure where practical instead of building an
unrelated storage system.

The V4 prepared layer should preserve:

- `science_state_id`, `nuisance_state_id`, and `render_state_id`;
- dataset family and split role;
- ordered physical and Fisher-scaled science vectors;
- ordered physical nuisance vectors;
- reproducible, content-addressable provenance;
- grouped science-state splits without leakage;
- deterministic frozen validation/test pair manifests;
- dynamic pair generation and future pair-family mixtures;
- a path for V3 regression evaluation.

This layer is future work. The raw V4 render corpus is complete, but V4
training shards, pair curricula, and frozen V4 pair manifests are not yet
implemented.

## 10. Train/validation/test partitioning

Avoid random image-level splits when related states or nuisance replicas can leak across sets.

Maintain multiple explicit evaluation questions.

### Nuisance generalization split

V4 contains ten fixed nuisance states fully crossed with every science state.
The development contract must explicitly choose whether to:

- train on all ten nuisance states and evaluate robustness over cross-nuisance
  combinations; or
- reserve one or two nuisance identities globally during development to test
  generalization to nuisance states not observed in training.

Do not silently resolve this. The choice should be made before freezing the V4
split/evaluation contract. A final production model may eventually train on all
ten nuisance states after development decisions are frozen.

### Science-state interpolation/generalization split

Hold out physical grid locations or generated physical states independently of nuisance realization.

### Joint-state test

Use sparse-mixture or V4 full-dimensional states with several nonzero parameters to test whether a model trained on pair-grid structure generalizes beyond sparse two-axis perturbations.

### Noise generalization test

Use fixed, independently seeded noisy validation/test realizations while allowing dynamic noise augmentation during training.

### Eigenmode-stratified test

Tag test samples by projection onto strong/weak eigen-directions and report performance by eigenvalue/eigenmode group.

### Final locked test set

After model/loss decisions stabilize, generate or reserve an independently sampled nuisance/physical-state test set that is not used for iterative model selection.

The next phase should keep separate evaluation axes for science-state
generalization, Fisher-distance/capture radius, nuisance robustness, and
pair-family behavior.

## 11. Noise strategy

Keep canonical rendered images noiseless when possible and apply detector/observation noise dynamically during training.

Advantages:

- unlimited noise realizations without multiplying storage;
- easy control over training SNR/noise regime;
- clean separation of forward-model state and stochastic realization;
- natural positive pairs for noise-invariance training.

Evaluation should use deterministic fixed-noise seeds for reproducibility.

If the noise/variance model changes substantially across observations, consider providing a variance/noise map as an auxiliary input channel in a later experiment. Do not add this initially unless needed.

## 12. V4 completed raw corpus and pair curriculum direction

### 12.1 Completed corpus

V4 is no longer only a planning target. The Gattaca2 production render for
`shera_ml_master_v4` completed and passed audit on 2026-09-08. The canonical
durable raw root is:

```text
/projects/shera_hpc/data/ml_training/shera_ml_master_v4
```

The render used repository source snapshot
`3da21e603c779377b559c9b86182f7150bd33366`. Slurm array job 19450239 rendered
54 tasks (`0-53%32`) with 20,000 renders per full task, 4,960 renders in the
final task, 4 CPUs per task, 4 GB requested per task, 2 hour walltime, and
concurrency cap 32. Every task completed with `ExitCode 0:0`. Observed MaxRSS
was approximately 0.60 GB for ordinary full tasks and approximately 0.58 GB for
the final partial task; the 4 GB value was a request, not measured usage.

The task-summary audit reported `V4_TASK_SUMMARY_AUDIT: PASS`: 54 summary
files, task IDs 0 through 53, no missing or unexpected tasks, 1,064,960
attempted, rendered, and accounted renders, no failures, and exact range
coverage through final stop 1,064,960.

The filesystem audit found 1,064,960 FITS files and 1,064,960 JSON sidecars,
with matching totals. The measured corpus footprint is 123 GB. After
completion, `/projects/shera_hpc` reported 3.2 TB size, 878 GB used, 2.3 TB
available, and 28% utilization.

The canonical raw V4 corpus is frozen and complete. Do not delete or rewrite
it outside an explicit audited repair task.

### 12.2 Frozen scientific structure

The frozen V4 structure contains 106,496 science states, ten nuisance states,
and 1,064,960 full cross-product renders. Every science state is rendered
against all ten fixed nuisance states.

| family | train science | validation science | test science | total science |
| --- | ---: | ---: | ---: | ---: |
| `joint_full_v4` | 65,536 | 8,192 | 8,192 | 81,920 |
| `radial_capture_v4` | 16,384 | 4,096 | 4,096 | 24,576 |
| total | 81,920 | 12,288 | 12,288 | 106,496 |

`joint_full_v4` provides broad multivariate science-state coverage and should
act as the general high-dimensional training distribution. `radial_capture_v4`
provides controlled Fisher-distance/capture-radius coverage for
distance-balanced evaluation, curriculum design, and capture-range diagnostics.
Do not collapse these families into a single undifferentiated population
without preserving family metadata.

### 12.3 Recommended V4 pair families

The full science-by-nuisance crossing enables future controlled pair families:

- **A: same nuisance, different science.** This is the closest V4 analogue of
  the S01/S05 task. It preserves benchmark continuity and learns science
  correction without nuisance mismatch.
- **B: different nuisance, same science.** Target science correction is zero.
  This trains explicit registration-nuisance invariance and provides a strong
  same-science consistency constraint.
- **C: different nuisance, different science.** This learns science correction
  in the presence of registration mismatch and approximates the intended robust
  initializer use case.
- **Optional identity pairs.** Same science and same nuisance with zero target;
  use at low or controlled weight if useful for consistency.

These are recommended training-design directions, not completed studies. Do
not exhaustively materialize all combinatorial pairs. Prefer dynamic pair
generation for training and frozen deterministic manifests for validation and
test.

## 13. Capture-range and ADORA benchmark

The central scientific justification for ML should be tested explicitly.

### 13.1 Linearization-error screen

At a reference state compute the image Jacobian \(J\). For a state displacement \(\Delta z\), compare

\[
\Delta I_{\mathrm{exact}}
=I(z_0+\Delta z)-I(z_0)
\]

with

\[
\Delta I_{\mathrm{linear}}=J\Delta z.
\]

Define a weighted relative linearization error such as

\[
\epsilon_{\mathrm{lin}}
=
\frac{\|\Delta I_{\mathrm{exact}}-J\Delta z\|_W}
{\|\Delta I_{\mathrm{exact}}\|_W}.
\]

This cheaply maps where the forward model becomes nonlinear.

### 13.2 One-step linear/Fisher correction

For the same cases, evaluate the local linear/Gauss--Newton/Fisher correction and record:

- parameter correction error;
- cosine/alignment with the true correction;
- remaining Fisher-scaled state error;
- image-residual improvement.

### 13.3 ADORA convergence study

Run actual ADORA on a representative subset spanning:

- linear regime;
- weakly nonlinear regime;
- strongly nonlinear regime;
- apparent failure/capture boundary.

Compare:

1. ADORA from the original initial state;
2. linear/Fisher correction followed by ADORA;
3. Siamese-ML correction followed by ADORA.

Headline metric:

\[
P(\mathrm{ADORA\ convergence})
\quad\text{vs. initial mismatch/capture regime}.
\]

Also track iterations/runtime to convergence and final physical parameter errors.

### 13.4 Do not reduce capture range to one scalar only

A Fisher-scaled radius

\[
r=\|\Delta z\|_2
\]

is useful but not sufficient. Retain direction/family labels because equal-radius perturbations can have very different nonlinear behavior and degeneracy structure.

## 14. Experimental model sequence

Keep the initial model program intentionally small.

### M0: existing baselines

- current classifier results for context only;
- handcrafted features as a diagnostic/easy-regime baseline;
- local linear/Fisher estimator as the principal physics baseline.

### M1: Siamese supervised regression

Shared compact CNN encoder plus comparison MLP.

Loss:

\[
\mathcal L=\mathrm{MSE}(\hat{\Delta z},\Delta z).
\]

No contrastive/invariance auxiliary loss initially.

### M2: nuisance multitask

Add a nuisance correction head and compare against M1 under grouped nuisance splits.

### M3: embedding invariance

Add same-science/different-nuisance and/or same-state/different-noise representation consistency. Evaluate whether held-out nuisance/noise generalization actually improves.

### M4: eigenmode-aware loss

Keep physical/Fisher-scaled outputs and add controlled eigenmode weighting experiments:

- unweighted;
- strong-mode/Fisher weighted;
- weak-mode emphasized with flooring/capping.

### M5: noisy training

Add dynamic observation noise and fixed-noise validation/test data.

### M6: V4 joint-state training

Train/evaluate on the prepared V4 corpus with explicit pair-family mixtures,
family metadata, grouped science-state splits, and nuisance-robust validation
manifests.

### M7: ADORA-in-the-loop evaluation

Use ML output only as an initializer/correction and measure capture-range benefit.

Further complexity such as iterative learned updates, uncertainty heads, adversarial nuisance removal, explicit JEPA predictors, or larger backbones should be deferred until these experiments establish a reason.

## 15. Notebook and code organization

Preserve the colleague's useful pattern of self-contained, well-scoped notebooks. A possible sequence is:

```text
01_dataset_audit_and_loader.ipynb
02_siamese_regression_baseline.ipynb
03_nuisance_multitask.ipynb
04_embedding_invariance.ipynb
05_eigenmode_aware_loss.ipynb
06_noise_robustness.ipynb
07_v4_joint_state.ipynb
08_adora_capture_range.ipynb
```

Reusable components should migrate into Python modules only after their interfaces stabilize:

- Parquet/shard dataset loader;
- pair sampler;
- shared CNN encoder;
- Siamese comparison head;
- loss helpers;
- metrics/eigenmode diagnostics;
- W&B logging wrapper.

Avoid building a large generalized ML framework before at least one end-to-end baseline works.

## 16. Experiment tracking

Weights & Biases is a strong candidate for run-level experiment tracking.

Track at minimum:

- git commit / notebook version;
- dataset/preprocessed-dataset version;
- split definition;
- parameter/FIM/eigenbasis IDs;
- shard dtype;
- model configuration;
- nuisance/invariance settings;
- optimizer/hyperparameters;
- random seeds;
- training/validation curves;
- per-parameter physical and Fisher-scaled errors;
- eigenmode-stratified errors;
- nuisance errors;
- capture-range diagnostics when available.

Use the cluster/project filesystem as the authoritative store for large datasets. Do not automatically upload the bulk FITS/shard corpus to a hosted experiment tracker. Need to confirm whether this is worth paying for.

## 17. Core metrics

Do not use one aggregate validation MSE as the sole performance measure.

Report:

- per-parameter bias, RMSE, MAE in native physical units;
- per-parameter error in Fisher-sigma units;
- aggregate Fisher-scaled state norm error;
- eigenmode-coordinate RMSE vs eigenvalue/rank;
- science and nuisance-head errors separately;
- performance vs nuisance realization;
- performance vs noise/SNR;
- performance vs Fisher-scaled mismatch radius;
- performance vs M1/M2 WFE norm;
- image residual before/after predicted update;
- comparison to one-step linear estimator;
- eventual ADORA convergence success and iterations/runtime.

For weak-mode-weighted experiments, always report ordinary physical/Fisher-scaled metrics in addition to the weighted training objective so improvements cannot be hidden by the choice of loss.

## 18. Immediate implementation sequence

### Phase A: controlled V3 bridge

1. Define, but do not launch here, the controlled bridge experiment combining
   the `S05-E04` larger `concat_diff` architecture with the `S01-E03`
   fixed-1e-3 longer-training prescription on the frozen V3 benchmark.
2. Run seed 11 first when the experiment is explicitly authorized.
3. If seed 11 is clearly promising, repeat the same combined prescription at
   seeds 23 and 47.
4. Keep the frozen V3 test set locked during model selection.

### Phase B: V4 prepared-dataset layer

1. Preserve the raw V4 FITS + JSON corpus as the authoritative source.
2. Build efficient training arrays or shards plus inspectable index/schema
   metadata.
3. Preserve science, nuisance, render, family, split, vector-space, and content
   identities.
4. Validate dtype conversion, shape, finite values, source round trips, and
   corpus accounting.
5. Support dynamic pair generation, pair-family mixtures, grouped science-state
   splits, deterministic frozen validation/test pair manifests, and V3
   regression evaluation.

### Phase C: V4 split/evaluation contract

1. Resolve the nuisance-generalization question before freezing splits.
2. Define separate evaluation axes for science-state generalization,
   Fisher-distance/capture radius, nuisance robustness, and pair-family
   behavior.
3. Preserve per-parameter Fisher-scaled metrics, overall Fisher RMSE, MSE
   skill, correction-vector geometry, distance-bin diagnostics, and locked
   test-set discipline.
4. Reuse the existing campaign analyzer conventions for normalized run tables,
   prediction geometry, per-parameter metrics, slices, and distance bins.

### Phase D: V4 training curriculum

1. Start from same-nuisance/different-science pairs for S01/S05 continuity.
2. Add different-nuisance/same-science zero-target pairs for explicit nuisance
   consistency.
3. Add different-nuisance/different-science pairs for robust initializer
   training.
4. Use optional identity pairs at low or controlled weight only if they improve
   consistency without dominating the task.

### Phase E: capture-range and ADORA

1. Use `radial_capture_v4` for distance-balanced diagnostics and curricula.
2. Compare ADORA alone, linear initialization + ADORA, and ML initialization +
   ADORA after model-selection decisions are frozen.

## 19. Open decisions

The following should remain explicit rather than silently resolved:

- exact physical science parameter set for the first Siamese model;
- whether nuisance variables are excluded from the science FIM before/after Schur reduction;
- exact Fisher-sigma reference point and whether scales need refreshing for future datasets;
- primary science eigenbasis: likely nuisance-marginalized Schur, but confirm;
- encoder embedding dimension and whether spatial attention is retained in baseline v1;
- exact image normalization that preserves perturbation amplitude and photometric information;
- default shard size and dtype after cluster benchmarks;
- float32 precision acceptance thresholds;
- whether V4 development trains on all ten nuisance states or reserves one or
  two nuisance identities globally for held-out nuisance generalization;
- the exact V4 pair-family mixture and relative weights;
- the frozen V4 validation/test pair-manifest recipes;
- whether the `S05-E04` architecture combined with the `S01-E03` optimizer
  prescription clears the V3 bridge at seed 11;
- eigenvalue flooring/capping for weak-mode loss weights;
- whether W&B/cloud artifact upload is institutionally acceptable and, if so, which artifacts may be uploaded.

## 20. Current recommendation summary

The current preferred path is:

\[
\boxed{
[I_{\mathrm{model}},I_{\mathrm{obs}}]
\xrightarrow{\text{shared encoder}}
\widehat{\Delta z}_{\mathrm{science}}
}
\]

with optional nuisance prediction and later embedding-invariance losses. The
output remains in interpretable Fisher-scaled physical coordinates. Eigenmodes
are used primarily to diagnose and deliberately weight difficult directions
rather than replacing physical outputs. The completed V3 benchmark results give
the regression baseline and candidate architecture/training choices; the
completed V4 raw corpus supplies the next main data source once prepared
training shards and frozen pair manifests exist. The decisive scientific
benchmark remains whether the learned correction expands the state-space region
from which ADORA reliably converges compared with the existing local
linear/Fisher initializer.

## 21. Relevant repository/workflow references

Current implementation/docs to consult while turning this plan into code:

- `docs/dev/ml_training_dataset_v2.md`
- `docs/dev/ml_prepared_dataset_wave1.md`
- `docs/dev/notes/ml_program_status_20260908.md`
- `work/experiments/ml/datasets/README.md`
- `work/experiments/ml/datasets/master_v4_spec.md`
- `work/experiments/generate_training_dataset_v3.py`
- `work/experiments/generate_training_dataset_v3_template.yaml`
- `tests/test_export_training_dataset_eigenmodes.py`
- `examples/recipes/canonical_astrometry.py`
- `examples/recipes/prescribed_monte_carlo.py`
- `docs/architecture/eigenmodes.md` when available in the active checkout
- `docs/dev/AGENTS.md`
- colleague notebooks reviewed during this planning cycle:
  - `SHERA_pair_grid_two_parameter_detection_with_handcrafted_updated.ipynb`
  - `SHERA_pair_grid_nuisance_comparison.ipynb`
  - `eigenmodes_test.ipynb`
  - `SHERA_eigenmode_degeneracy_comparison.ipynb`


## 22. ML Experiment Program Status

This section is the curated scientific ledger for model experiments. It should
summarize study intent, fixed artifacts, and headline outcomes without becoming
a run-by-run machine log. Future W&B integration should live at the Run level.

### 22.1 Permanent nomenclature

- **Study:** broad scientific research question, e.g. `S01`.
- **Experiment:** controlled test inside a Study, e.g. `S01-E01`.
- **Run:** one concrete training execution, e.g. `S01-E01-R001`.
- **Shared artifact:** reusable versioned object, e.g. `PREP-V3-v1`,
  `SPLIT-ML-v1`, `PAIR-EVAL-v1`, or future `LIN-EVAL-v1`.

Ordered image pairs always use:

\[
A=\text{current/reference/model state}, \qquad
B=\text{target/observation state},
\]

with supervised target:

\[
\Delta z_{\mathrm{science}} = z_B-z_A.
\]

### 22.2 Study registry

| Study | name | status | notes |
|---|---|---|---|
| S01 | Pairwise Correction Learnability | completed V3 baseline + optimizer wave | Three-seed baseline and seed-11 optimizer/training-control wave completed on LS6; ordinary model-selection results are validation results. |
| S02 | Registration Nuisance Robustness | provisional/planned | Relax same-nuisance pairing and measure robustness to registration changes. |
| S03 | Observation-Noise Robustness | provisional/planned | Enable dynamic observation noise and fixed noisy eval manifests. |
| S04 | Learned vs Local-Linear Correction | provisional/planned | Compare ML corrections with Binder/Jacobian/Fisher linear evaluation. |
| S05 | Architecture / Representation Study | completed Wave 1 | Seed-11 architecture wave completed on LS6; `S05-E04` is the provisional architecture winner. |
| S06 | V3 Architecture / Training Closure | repository-prepared | Combines the S05 architecture winner with the S01-E03 training-control candidate on the frozen V3 contract; 9 planned runs. |
| S07 | V4 Entry and Data Scaling | repository-prepared | Brings up `joint_full_v4` A-pair training, shared V4 scaler use, and joint-train prefix scaling; 10 planned runs. |
| S08 | V4 Nuisance Robustness | repository-prepared | Tests C-only, ABC, multitask nuisance prediction, and an unseen-nuisance holdout profile; 12 planned runs. |
| S09 | V4 Capture / Radial Curriculum | repository-prepared | Mixes broad `joint_full_v4` A pairs with distance-balanced `radial_capture_v4` A pairs and radial-only curriculum; 9 planned runs. |
| S10 | V4 Fisher/Eigenmode-Aware Loss | repository-prepared | Compares the S07 large clean objective with fixed-eigenbasis strong- and weak-mode weighted losses; 9 planned runs including the shared clean reference cohort. |
| S11 | V4 Pairwise Physics Consistency | repository-prepared | Adds explicit antisymmetry and low-weight identity auxiliary losses while referencing the S10-E01 clean cohort; 6 new planned runs. |
| S12 | V4 Photon-Noise Observation Robustness | repository-prepared | Uses dynamic physical photon noise before ML scaling, fixed photon-noise validation, and optional noise-consistency loss while referencing S10-E01; read noise and dark current are future extensions; 6 new planned runs. |

Deferred roadmap ideas that previously occupied the S06-S08 labels are retained
under non-conflicting future labels:

- `F01` Fisher / Eigenmode Structure: diagnose and possibly weight errors by
  Fisher/eigenmode structure.
- `F02` Learned vs Local-Linear Generalization Extensions: broaden the
  Binder/Jacobian/Fisher baseline comparisons after V4 training results exist.
- `F03` ADORA Initialization / Capture Range: use trained correction models as
  initializers inside ADORA-style nonlinear optimization and measure convergence
  behavior directly.

### 22.3 S01 record

**Study:** `S01` — Pairwise Correction Learnability

**Research question:** Can a simple shared-weight image encoder estimate
Fisher-scaled science-state corrections from noiseless image pairs under
controlled registration?

**Initial shared artifacts:**

- `PREP-V3-v1`: Wave 1 prepared sample-centric image/state store.  The
  `PREP-V3-nuisance-v1` path is the prepared dataset instance/root name, not
  the catalog artifact ID.  Current prepared hash:
  `4cdc325fbf8d4a0e07195ab075bea6f5035dfc01c9990cac03ee1f59c131e5e6`.
- `SPLIT-ML-v1`: reusable science-state and nuisance-realization split registry,
  pinned for S01 by stable content SHA256 as well as artifact ID.  Current
  content SHA256:
  `29f0e95c3819cbeb5ce00aafb593445510723ea5fc20e2e7f3e585c1b9615314`.
- `S01-VALIDATION-PAIRS-v1`: frozen validation ordered-pair manifest, 2048
  ordered pairs, content SHA256
  `68ccd41a35d286c8b060f291eef6c788a6b0d97c9660868f74e01b2b4feae499`.
- `S01-TEST-PAIRS-v1`: frozen test ordered-pair manifest, 4096 ordered pairs,
  content SHA256
  `375451064bd363a6afb33c6f3f1bdff7e92efe1384c1513b3491b42318c87b82`.

**Initial experiments:**

- `S01-E00` — Pipeline / tiny-overfit sanity.
- `S01-E01` — Clean same-nuisance held-out science regression.
- `S01-E02` through `S01-E07` — seed-11 optimizer/training-control wave.

| ID | research objective | pair policy | nuisance policy | noise policy | split artifact | model/config | status | headline result | notes |
|---|---|---|---|---|---|---|---|---|---|
| S01-E00 | Verify image loading, target construction, shared encoder, gradients, checkpointing, and metrics end-to-end. | Same nuisance, different science; tiny deterministic development pairs; reverse pairs available. | Training nuisance partition only. | Off. | `SPLIT-ML-v1` | Small shared CNN, `concat_diff`, MSE on `z_B-z_A`. | implemented / pending real-data run | Pending. | Success criterion is substantial overfit of a tiny noiseless set; not a generalization result. |
| S01-E01 | Measure clean held-out science correction regression under fixed registration within each pair. | Same V3 pair-grid where available, same nuisance, different science, configurable Fisher-distance range. | Evaluate both held-out science with train-seen nuisance and held-out science with held-out nuisance. | Off. | `SPLIT-ML-v1` + frozen S01 validation/test pairs | Shared CNN, default `concat_diff`, AdamW. | completed on LS6 | Three-seed mean best validation RMSE 74.1839, sample SD 1.6486, versus zero-correction Fisher RMSE 250.840. | Validation result only; `evaluate_test: false`. This is not yet a nuisance-invariance study because nuisance is fixed inside each pair. |
| S01-E02..E07 | Test optimizer/training-control changes while preserving the S01-E01 architecture and evaluation contract. | Same as S01-E01. | Same as S01-E01. | Off. | `SPLIT-ML-v1` + frozen S01 validation/test pairs | Same model; fixed, plateau, and cosine LR variants. | completed on LS6 | `S01-E03` is current training-control candidate, best validation RMSE 57.8676. | One seed only; late best checkpoints mean convergence is not proven. |

Production baseline replicas completed on Lonestar6:

| run | seed | LS6 job | best validation RMSE | best epoch | MSE skill |
|---|---:|---:|---:|---:|---:|
| `S01-E01-R001` | 11 | 3418678 | 72.4681 | 99 / 100 | 0.916535 |
| `S01-E01-R002` | 23 | 3418707 | 75.7559 | 94 / 100 | 0.908790 |
| `S01-E01-R003` | 47 | 3418708 | 74.3278 | 93 / 100 | 0.912197 |

Optimizer/training-control wave:

| run | initial LR | scheduler | epochs completed | best epoch | best validation RMSE | MSE skill |
|---|---:|---|---:|---:|---:|---:|
| `S01-E02-R001` | 5e-4 | fixed | 300 | 278 | 59.5046 | 0.943726 |
| `S01-E03-R001` | 1e-3 | fixed | 300 | 273 | 57.8676 | 0.946780 |
| `S01-E04-R001` | 5e-4 | reduce-on-plateau | 242 | 219 | 64.9824 | 0.932888 |
| `S01-E05-R001` | 1e-3 | reduce-on-plateau | 229 | 222 | 67.0941 | 0.928455 |
| `S01-E06-R001` | 5e-4 | cosine | 300 | 278 | 60.3250 | 0.942163 |
| `S01-E07-R001` | 1e-3 | cosine | 300 | 275 | 60.3282 | 0.942157 |

`S01-E03` reduces best validation RMSE by approximately 20.1% relative to
`S01-E01-R001`. A fixed 1e-3 learning rate with longer training outperformed
the tested plateau and cosine prescriptions in this wave. Do not claim 300
epochs proves convergence.

### 22.4 S05 Wave 1 architecture study

`work/experiments/ml/s05/` tracks the first architecture/representation study.
It reuses the exact S01 prepared dataset, split registry, pair policy, frozen
validation pairs, frozen test pairs, image scaling, no-noise condition,
optimizer, learning rate, batch size, pairs per epoch, and early-stopping
policy. First-pass variants all use seed 11 so architecture differences are
compared under one common deterministic training seed and pair stream.

| ID | change from S05-E01 | model/config | best validation RMSE | status |
|---|---|---|---:|---|
| `S05-E01` | Reference baseline matching S01-E01 seed-11 prescription except identity fields. | `[16, 32, 64, 128]`, embedding 128, encoder/head 256, `concat_diff`, approximately 767k parameters. | 72.3262 | completed |
| `S05-E02` | Comparator only. | Same capacity as E01, `difference` comparator. | 89.2171 | completed |
| `S05-E03` | Coordinated smaller-capacity bracket. | `[8, 16, 32, 64]`, embedding 64, encoder/head 128, `concat_diff`. | 89.3951 | completed |
| `S05-E04` | Coordinated larger-capacity bracket. | `[32, 64, 128, 256]`, embedding 256, encoder/head 512, `concat_diff`, approximately 3.055M parameters. | 68.4548 | completed / provisional winner |

`S05-E01` reproduces the `S01-E01` seed-11 baseline closely. Difference-only
comparison and the smaller model are substantially worse than baseline. The
larger `S05-E04` model is the best Wave 1 architecture result and improves
best validation RMSE by approximately 5.35% relative to `S05-E01`. It remains
provisional because it has only one production seed, used the old 5e-4 /
100-epoch training prescription, and has not been combined with the `S01-E03`
optimizer/training-control candidate. Do not use the frozen test set for model
selection.

### 22.5 V4 raw corpus status

The V4 raw corpus `shera_ml_master_v4` is complete and audited at
`/projects/shera_hpc/data/ml_training/shera_ml_master_v4`. It was rendered on
Gattaca2 from source snapshot `3da21e603c779377b559c9b86182f7150bd33366` under
Slurm array job 19450239 (`0-53%32`). All 54 tasks completed with
`ExitCode 0:0`, and the task-summary audit reported
`V4_TASK_SUMMARY_AUDIT: PASS` with 1,064,960 attempted, rendered, and
accounted renders.

Filesystem audit found 1,064,960 FITS files and 1,064,960 JSON sidecars across
`joint_full_v4` and `radial_capture_v4`, with matching totals and a measured
corpus footprint of 123 GB. Every V4 science state is rendered against all ten
fixed nuisance states, enabling future controlled pair families without
rewriting the raw corpus.

Detailed execution evidence is recorded in
`docs/dev/notes/ml_program_status_20260908.md` and
`work/experiments/ml/datasets/README.md`.

### 22.6 Split and pair artifact semantics

The prepared dataset remains authoritative and sample-centric: one prepared row
is one rendered image/state. ML pairs are references into that store, not copied
image arrays. Pair manifests carry stable `pair_record_id`, `sample_a_id`, and
`sample_b_id` keys so future physics baselines can join predictions to the same
ordered pairs.

PyTorch is an optional ML-layer dependency, not a core `dluxshera` import
requirement. Use the project ML extra, e.g. `python -m pip install -e .[ml]`,
before running the CNN model or training CLI.

Science-state splitting uses the prepared physical-delta identity
(`group_ids.physical_delta_sha256`) rather than `pair_id`, grid cell, nuisance
ID, or filenames. Nuisance realization splitting is recorded separately and may
use explicit assignments or deterministic fraction-based assignments. Pair
generation happens after these state-level splits.

For production S01 runs, `train_from_study.py` enforces the same study contract
as GPU preflight before training starts: prepared catalog identity, split
artifact ID and content hash, frozen validation recipe, optional frozen test
recipe, and resolved experiment pair policy.  Resume is same-run continuation:
existing history is appended, prior best artifacts remain valid if the resumed
segment does not improve, and changed scientific identities are rejected.

Small study-defining S01 artifacts should survive scratch cleanup.  Keep working
copies on `/scratch-jpl` or `/scratch-edge`, and retain durable split,
validation-pair, test-pair, and compact provenance artifacts under
`/projects/shera_hpc/$USER/dLuxShera-Results/ml/S01/artifacts/`.  Do not copy
the large prepared shard store into that results artifact tree when a canonical
prepared-data copy already exists elsewhere under `/projects`.

Dynamic training datasets interpret `pairs_per_epoch` as the total number of
ordered examples. When `include_reverse=True`, adjacent examples are generated
from one valid sampled base pair: index `2k` returns `(A, B)` with
`target_delta_z = z_B - z_A`, and index `2k+1` returns `(B, A)` with the
negated science, physical, and nuisance deltas. Thus `pairs_per_epoch=2048`
means 1024 sampled base pairs and 2048 optimizer examples, not a doubled epoch.
Observation-only dynamic noise is applied after ordering and remains attached
to the B role.

Recommended configurable layout:

```text
<scratch>/ml_data/
  PREP-V3-v1/
  catalogs/

<project/results>/ml_experiments/
  splits/
  pair_manifests/
  S01/
    S01-E00/
    S01-E01/
```

### 22.7 Future local-linear evaluation convention

The future local-linear physics baseline should use the ordered pair convention
above. For pair `(A, B)`, define `r_AB = I_B - I_A` and evaluate the image
Jacobian at the reference state `A`:

\[
J_A = \left.\frac{dI}{d\theta}\right|_{\theta_A}.
\]

For the initial shot-noise-dominated convention, use a numerically safe variance
model derived from the noiseless reference image `A`, matching the existing
canonical/Fisher estimator where practical. Then:

\[
F_A = J_A^T W_A J_A, \qquad
g_{AB} = J_A^T W_A (I_B-I_A),
\]

and:

\[
\Delta\theta_{\mathrm{linear}} = F_A^{-1}g_{AB}.
\]

Under this convention, `J_A`, `W_A`, and `F_A` depend only on the reference
state `A`; `B` enters through the residual. Multiple `B` targets can therefore
reuse the same `A`-state linearization/Fisher information. This should be a
future evaluation artifact keyed by `pair_record_id`, not a model-training
input. Do not confuse this Gauss-Newton/Fisher baseline with the exact nonlinear
loss Hessian, which may contain residual-dependent second-order terms.

### 22.8 Follow-up: nominal V3 Fisher artifact

The V3 generator already computes a nominal FIM to derive Fisher-diagonal
parameter sigmas. A small future capability patch should preserve that already
computed nominal information without regenerating the dataset:

- reference theta/state;
- reference image or unambiguous reference-image identity;
- full nominal FIM;
- FIM parameter labels;
- Fisher sigmas;
- variance/weighting convention;
- variance floor or low-count handling;
- exposure/count normalization needed to reproduce the FIM.

Do not add per-sample Jacobians, FIMs, or Hessians to the prepared image store.
Those objects are large and should be generated later only as explicit
evaluation artifacts for selected reference states.
# S06-S09 Training Addendum

The S08-E03 multitask path keeps the existing shared Siamese encoder and
science-correction head. It adds an optional nuisance-delta head for
registration x, registration y, and roll. Science remains the primary task.

When enabled, the nuisance target is divided by a fixed component scale derived
from the training nuisance bank unless an explicit positive scale vector is
provided. The recorded loss is:

```text
loss_total = loss_science + lambda_nuisance * loss_nuisance
```

`lambda_nuisance` defaults to 1.0. Checkpoint selection for S08-E03 uses
science validation loss, not total multitask loss. Prediction products retain
`pair_record_id` and include nuisance prediction arrays for multitask runs.

S09 capture diagnostics report Fisher RMSE, MSE skill, correction-vector
alignment, initial and remaining Fisher distances, `rho = d1 / max(d0, eps)`,
fractions with `rho < 1`, rho quantiles, metrics by initial-distance bin, by
dataset family, and remaining-distance threshold fractions at 100, 250, 500,
1000, and 2000. These thresholds are generic diagnostics and are not labeled
as an ADORA capture radius.

# S10-S12 Training Addendum

The S10-S12 launch plan is tracked in
`docs/dev/notes/ml_s10_s12_campaign_plan.md`. The wave uses one fixed S07-large
V4 backbone so the changed variables are loss weighting, pairwise physical
consistency, and observation-noise robustness rather than architecture.

S10 keeps network outputs in canonical Fisher-scaled science coordinates and
projects prediction errors into a fixed, materialized science eigenbasis only
inside the loss and diagnostics. The S10-v1 eigenbasis artifact records the
source coordinate space, the prepared Fisher-scaled eigenbasis coordinate
space, parameter labels, PREP-V4 Fisher scales, physical-FIM identity,
transformed \(F_z\) identity, nominal provenance, and the weighting/variance
convention. S11 computes reversed and identity predictions from the current
loaded batch. S12 applies physical photon noise to count-space images before ML
scaling and uses an authenticated fixed photon-noise validation recipe for
deterministic model selection. Read noise, dark current, and a full detector
noise sweep remain future extensions.
