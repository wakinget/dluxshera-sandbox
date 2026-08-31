# SHERA ML Inverse-Model Design and Experiment Roadmap

**Status:** Working design note / first draft  
**Date:** 2026-08-25  
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

## 10. Train/validation/test partitioning

Avoid random image-level splits when related states or nuisance replicas can leak across sets.

Maintain multiple explicit evaluation questions.

### Nuisance generalization split

Hold out complete nuisance realizations from training. Example concept:

- train: majority of nuisance IDs;
- validation: one unseen nuisance ID;
- test: one or more additional unseen nuisance IDs.

The exact allocation should be recorded and may be changed depending on the available replicate count.

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

## 11. Noise strategy

Keep canonical rendered images noiseless when possible and apply detector/observation noise dynamically during training.

Advantages:

- unlimited noise realizations without multiplying storage;
- easy control over training SNR/noise regime;
- clean separation of forward-model state and stochastic realization;
- natural positive pairs for noise-invariance training.

Evaluation should use deterministic fixed-noise seeds for reproducibility.

If the noise/variance model changes substantially across observations, consider providing a variance/noise map as an auxiliary input channel in a later experiment. Do not add this initially unless needed.

## 12. V4 full-dimensional dataset generation

### 12.1 Motivation

Pair grids and sparse mixtures remain valuable controlled diagnostics but do not represent a fully joint inverse problem. V4 should add a distinct full-dimensional sampling family rather than changing V3 semantics.

### 12.2 Scrambled Sobol sampling

Use a scrambled Sobol sequence as the first full-dimensional space-filling sampler. It provides reproducible low-discrepancy coverage without claiming a scientifically correct prior distribution.

Conceptual V4 family:

```yaml
experiment:
  datasets:
    sobol_joint:
      enabled: true
      sampler: sobol
      scramble: true
      seed: ...
      n_samples: ...
```

Powers of two are convenient pilot sizes:

- 4096: implementation/capture-range pilot;
- 16384: intermediate training dataset;
- 65536: substantial first production set if justified.

### 12.3 Sampling regimes

Do not assume one enormous hypercube is the correct distribution. Support labeled regimes such as:

- local;
- intermediate;
- commissioning.

These labels describe intended capture-range regimes, not formal priors.

### 12.4 Commissioning-scale WFE constraints

Large independent per-Zernike bounds can create aggregate WFE far larger than the nominal coefficient scale. V4 planning should therefore track and optionally constrain:

- individual coefficient bounds;
- M1 low-order coefficient-vector norm/RMS;
- M2 low-order coefficient-vector norm/RMS;
- total mixed state radius in Fisher-scaled coordinates.

### 12.5 Nuisance sampling in V4

For bulk V4 data, sample one independent nuisance state per science state rather than repeating every science state over a fixed nuisance grid.

Additionally create a smaller anchor subset with repeated nuisance/noise realizations for the same science state. These anchors are especially valuable for invariance and multitask studies.

### 12.6 Eigenmode-focused companion data

Alongside space-filling Sobol draws, generate focused validation/stress sweeps along selected strong and weak eigen-directions. These should be separate named dataset families so their interpretation remains clear.

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

Train/evaluate on full-dimensional Sobol samples and commissioning-scale regimes.

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

### Phase A: persistent design and data audit

1. Review/refine this design note.
2. Audit the exact existing V3 nuisance dataset roots and metadata on the cluster.
3. Freeze a parameter-order/schema identifier and reference Fisher-sigma vector.
4. Define the preferred science eigenbasis provenance, likely nuisance-marginalized Schur after Fisher scaling, while retaining fixed/full bases as diagnostics.

### Phase B: preprocessing infrastructure

1. Implement a framework-agnostic V3/V4 preprocessor.
2. Build `index.parquet` and `schema.json`.
3. Convert source FITS into configurable `.npy` shards.
4. Default to float32 working shards but run/record float64-to-float32 precision audits.
5. Validate random shard samples against source FITS.
6. Benchmark loader memory, I/O throughput, and address-space behavior using all nuisance draws.

### Phase C: first learning experiments

1. Build one self-contained PyTorch Siamese-regression notebook using the colleague's compact CNN style.
2. Train on controlled same/different nuisance pair categories.
3. Add grouped validation/test splits and best-validation checkpoint selection.
4. Benchmark against the local linear/Fisher correction.
5. Add nuisance multitask head.
6. Add representation-invariance loss only after the multitask baseline is understood.

### Phase D: eigenmode experiments

1. Compute/document the scaled fixed/full/Schur eigensystems.
2. Add eigenmode error diagnostics to every run.
3. Test one or two controlled eigenmode loss-weighting choices.
4. Build focused weak-mode stress/validation sweeps if needed.

### Phase E: capture-range and V4

1. Run cheap linearization/one-step capture-range studies to inform useful state bounds.
2. Implement V4 scrambled-Sobol joint-state plan generation.
3. Render a small pilot (e.g. 4096 states).
4. Expand only if the pilot covers meaningful regimes and the learned model demonstrates value.
5. Compare ADORA alone, linear initialization + ADORA, and ML initialization + ADORA.

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
- train/validation/test nuisance-ID assignment;
- V4 local/intermediate/commissioning bounds;
- aggregate WFE constraints for Sobol sampling;
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

with optional nuisance prediction and later embedding-invariance losses. The output remains in interpretable Fisher-scaled physical coordinates. Eigenmodes are used primarily to diagnose and deliberately weight difficult directions rather than replacing physical outputs. The existing V3 nuisance dataset is sufficient to test the first Siamese/invariance ideas once the memory/I/O path is improved. Full-dimensional scrambled-Sobol V4 data then extend the model into the joint nonlinear/commissioning regime. The decisive scientific benchmark is whether the learned correction expands the state-space region from which ADORA reliably converges compared with the existing local linear/Fisher initializer.

## 21. Relevant repository/workflow references

Current implementation/docs to consult while turning this plan into code:

- `docs/dev/ml_training_dataset_v2.md`
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

- **Study:** broad scientific research question, e.g. `ML-S01`.
- **Experiment:** controlled test inside a Study, e.g. `ML-S01-E01`.
- **Run:** one concrete training execution, e.g. `ML-S01-E01-R001`.
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
| ML-S01 | Pairwise Correction Learnability | active | First shared-CNN regression substrate and clean pair baseline. |
| ML-S02 | Registration Nuisance Robustness | provisional/planned | Relax same-nuisance pairing and measure robustness to registration changes. |
| ML-S03 | Observation-Noise Robustness | provisional/planned | Enable dynamic observation noise and fixed noisy eval manifests. |
| ML-S04 | Learned vs Local-Linear Correction | provisional/planned | Compare ML corrections with Binder/Jacobian/Fisher linear evaluation. |
| ML-S05 | Architecture / Representation Study | provisional/planned | Controlled architecture and comparator changes after S01 baseline. |
| ML-S06 | Fisher / Eigenmode Structure | provisional/planned | Diagnose and possibly weight errors by Fisher/eigenmode structure. |
| ML-S07 | Joint-State Generalization | provisional/planned | Move beyond sparse pair-grid structure toward joint-state samples. |
| ML-S08 | ADORA Initialization / Capture Range | provisional/planned | Test whether learned corrections expand ADORA convergence capture range. |

### 22.3 ML-S01 record

**Study:** `ML-S01` — Pairwise Correction Learnability

**Research question:** Can a simple shared-weight image encoder estimate
Fisher-scaled science-state corrections from noiseless image pairs under
controlled registration?

**Initial shared artifacts:**

- `PREP-V3-v1`: Wave 1 prepared sample-centric image/state store.
- `SPLIT-ML-v1`: reusable science-state and nuisance-realization split registry.
- `PAIR-EVAL-v1`: frozen ordered-pair evaluation manifests.

**Initial experiments:**

- `ML-S01-E00` — Pipeline / tiny-overfit sanity.
- `ML-S01-E01` — Clean same-nuisance held-out science regression.
- `ML-S01-E02` — Comparator representation ablation.

| ID | research objective | pair policy | nuisance policy | noise policy | split artifact | model/config | status | headline result | notes |
|---|---|---|---|---|---|---|---|---|---|
| ML-S01-E00 | Verify image loading, target construction, shared encoder, gradients, checkpointing, and metrics end-to-end. | Same nuisance, different science; tiny deterministic development pairs; reverse pairs available. | Training nuisance partition only. | Off. | `SPLIT-ML-v1` | Small shared CNN, `concat_diff`, MSE on `z_B-z_A`. | implemented / pending real-data run | Pending. | Success criterion is substantial overfit of a tiny noiseless set; not a generalization result. |
| ML-S01-E01 | Measure clean held-out science correction regression under fixed registration within each pair. | Same V3 pair-grid where available, same nuisance, different science, configurable Fisher-distance range. | Evaluate both held-out science with train-seen nuisance and held-out science with held-out nuisance. | Off. | `SPLIT-ML-v1` + `PAIR-EVAL-v1` | Shared CNN, default `concat_diff`, AdamW. | implemented / ready to run | Pending. | This is not yet a nuisance-invariance study; nuisance is fixed inside each pair. |
| ML-S01-E02 | Compare whether absolute-state context improves correction regression. | Same as E01. | Same as E01. | Off. | `SPLIT-ML-v1` + `PAIR-EVAL-v1` | Switch comparator between `concat_diff` and `difference`. | ready / not launched | Pending. | No new architecture required. |

### 22.4 Split and pair artifact semantics

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

Recommended configurable layout:

```text
<scratch>/ml_data/
  PREP-V3-v1/
  catalogs/

<project/results>/ml_experiments/
  splits/
  pair_manifests/
  ML-S01/
    ML-S01-E00/
    ML-S01-E01/
```

### 22.5 Future local-linear evaluation convention

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

### 22.6 Follow-up: nominal V3 Fisher artifact

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
