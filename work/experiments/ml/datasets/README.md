# ML Dataset Planning

This directory holds planning and read-only inspection tooling for SHERA ML
rendered-state datasets. It is intentionally separate from training studies:
the tools here audit dataset contracts and support V4 state-plan design, but do
not launch render jobs or mutate historical artifacts.

## Audit Utility

Run a read-only audit against a raw rendered dataset, compact context package,
or prepared ML dataset:

```bash
python3 work/experiments/ml/datasets/audit_dataset.py \
  /path/to/dataset-or-context \
  --output-json /tmp/dataset_audit.json
```

Optional split outputs:

```bash
python3 work/experiments/ml/datasets/audit_dataset.py \
  /path/to/dataset-or-context \
  --output-dir /tmp/dataset_audit
```

`--output-dir` writes:

- `audit_summary.json`
- `parameter_summary.csv`
- `nuisance_summary.csv`
- `sparse_summary.csv`
- `integrity_summary.json`

The audit is metadata-driven and streams JSONL rows. It does not load the full
image corpus. Use `--verify-files` only for full raw dataset audits when the
rendered FITS and per-sample JSON files are expected to be present:

```bash
python3 work/experiments/ml/datasets/audit_dataset.py \
  /path/to/full/raw/dataset \
  --verify-files \
  --output-json /tmp/full_dataset_audit.json
```

Compact historical context packages intentionally omit most rendered files, so
file-existence verification is opt-in.

## Raw Dataset Audit

Raw V3-style datasets are detected from:

- `manifest.json`
- `parameter_space.json`
- `samples.jsonl`
- optional `pair_plan.csv`
- optional `sparse_mixture_plan.csv`
- optional `prescription_resolved.json`

Example:

```bash
python3 work/experiments/ml/datasets/audit_dataset.py \
  "Results/ML Training Datasets/shera_training_dataset_nuisance_pairs_20260511" \
  --output-json /tmp/v3_nuisance_audit.json
```

## Compact Historical Context Audit

The local sparse 100k context package is enough to audit metadata and sampling
geometry without the full Gattaca2 FITS corpus:

```bash
python3 work/experiments/ml/datasets/audit_dataset.py \
  "Results/ML Training Datasets/shera_test_dataset_sparse_nuisance_20260707_context" \
  --output-json /tmp/sparse100k_audit.json
```

The matching full dataset audit on Gattaca2 should use the persistent source
path:

```bash
python3 work/experiments/ml/datasets/audit_dataset.py \
  /projects/shera_hpc/data/ml_training/shera_test_dataset_sparse_nuisance_20260707 \
  --output-json /tmp/sparse100k_full_audit.json
```

The audit must remain read-only. Do not modify historical dataset contents.

## Prepared Dataset Audit

Prepared V3 datasets are detected from:

- `manifest.json`
- `vector_spaces.json`
- `array_shards_manifest.json`
- `index.jsonl`
- `shards/*.npy`

Example:

```bash
python3 work/experiments/ml/datasets/audit_dataset.py \
  "Results/ML Training Datasets/preprocessed/PREP-V3-nuisance-v1" \
  --output-json /tmp/prep_v3_nuisance_audit.json
```

## Dataset-Family Policy

V3 structured pair-grid and nuisance-pair datasets remain useful for controlled
single-parameter signatures, pairwise interactions, optimizer comparisons, and
continuity with S01/S05.

The historical sparse 100k corpus is a first-class legacy dataset, but its
records are historically labeled `split = "test"`. Preserve that provenance.
Do not relabel those samples as production training data. Exploratory use must
be explicitly marked non-confirmatory.

V4 is a logical master collection, not a requirement to concatenate raw renders
into one directory. Future composite catalogs should select families from
canonical raw render roots without copying images.

## Projects And Scratch

Persistent canonical assets should live under Projects storage, conceptually:

```text
/projects/shera_hpc/data/ml_training/shera_ml_master_v4/
```

Use Projects for frozen prescriptions, immutable state plans, canonical renders,
render metadata, audits, and content hashes.

Use scratch for transient staging, JAX caches, shard retries, and conversion
intermediates, conceptually:

```text
/scratch-jpl/shera_hpc/$USER/dLuxShera-ML/render_v4/
```

Before a V4 render launch, measure current capacity on Gattaca2:

```bash
df -h /projects/shera_hpc
du -sh /projects/shera_hpc/data/ml_training/*
```

Do not claim current free space from stale notes.

## V4 Materialization

Deterministic V4 state-plan materialization and QA are implemented by:

```bash
python3 work/experiments/ml/datasets/materialize_master_v4.py \
  --outdir work/experiments/ml/datasets/materialized/master_v4
```

For a small development smoke run:

```bash
python3 work/experiments/ml/datasets/materialize_master_v4.py \
  --tiny \
  --outdir work/experiments/ml/datasets/materialized/master_v4_tiny
```

The materializer writes state plans, vector contracts, nuisance-bank
comparisons, QA summaries, a compact render-index contract, and handoff notes.
It does not render images or submit cluster jobs.

V4 science ordering is not inferred from raw `parameter_space.json` row order,
mapping order, or JSON serialization order. The canonical science order is the
stored `spaces.fisher_scaled_delta.components` order in the prepared S01/S05
V3 `vector_spaces.json`; raw parameter records are reconciled by label into
that order before any nominal vectors, Fisher scales, envelopes, state vectors,
compatibility tables, QA summaries, or IDs are generated. Missing, duplicate,
or unexpected science labels fail materialization.

V4 nuisance ordering is a separate explicit contract:
`source.x_position_as`, `source.y_position_as`,
`source.position_angle_deg`. The selected V3 nuisance-bank artifact must
contain exactly those labels, but its source-key order is not used as the
authoritative vector order. Nuisance values are written by label into the
canonical order.

Generated materializations under `work/experiments/ml/datasets/materialized/`
are intentionally ignored by Git. Track the materializer source, audit source,
tests, this README, and `master_v4_spec.md`; do not track generated JSONL state
plans, QA outputs, freeze manifests, render contracts, notebooks, or transfer
packages. If a tiny fixture is needed for a unit test, create a minimal fixture
under `tests/` rather than committing `master_v4_tiny`.

The materializer will not silently replace a populated output directory. A new
directory or an existing empty directory is allowed. A populated directory fails
unless replacement is explicit:

```bash
python3 work/experiments/ml/datasets/materialize_master_v4.py \
  --outdir work/experiments/ml/datasets/materialized/master_v4 \
  --overwrite
```

`--overwrite` removes and recreates the generated artifact tree before writing,
so a frozen materialization is not partially mixed with newly generated files.
Use `--dry-run` to check the output-root policy without writing artifacts.

The generated `qa/split_integrity_summary.json` validates science IDs and
canonical ordered physical science vectors across `joint_full_v4` and
`radial_capture_v4`, across train/validation/test, and across family
boundaries. V4 materialization fails on duplicate new-V4 science IDs, duplicate
new-V4 physical science vectors, same-family cross-split leakage, cross-family
overlaps, or any new-V4 train/test physical-vector collision. Recoverable
overlap with historical V3/legacy datasets is reported separately when audited
and is not a V4 materialization failure.

Scientific hashes are path-independent. Ordered vectors, independent science
and nuisance vector-space identities, sampling envelopes, seeds, counts,
nuisance vectors, render-system content, render indexing, and review decisions
participate in scientific identity. Local source paths, output roots,
timestamps, hostnames, and temporary staging paths are provenance only.

`science_state_id` depends on the dataset/family identity, split role, the
science vector-space ID, the canonical ordered physical science vector, the
canonical ordered Fisher vector, and the sampling-family contract. It is not a
seed, sequence-index, or file-position identity. `nuisance_state_id` depends on
the nuisance vector-space ID and canonical ordered physical nuisance vector.
`render_state_id` depends only on `science_state_id`, `nuisance_state_id`, and
`render_system_contract_hash`.

`render_system_contract.json` is built from the authoritative resolved S01/S05
compatible `system` subtree and noise policy. Scientifically relevant file
references are identified by content hash when available; local file paths do
not affect the render-system scientific hash. `render_contract.json` defines the
compact full-cross-product mapping:

```text
render_index = science_global_index * nuisance_count + nuisance_bank_index
```

Family order is `joint_full_v4`, then `radial_capture_v4`; split order is
`train`, `validation`, then `test`. The compact contract maps each render index
to the family, split, science-plan row/global index, science state ID, nuisance
bank index, nuisance state ID, and render state ID formula without expanding the
canonical plan to one row per render.

Nuisance-bank rows include stable `nuisance_state_id` values derived from the
nuisance vector-space scientific identity and canonical ordered physical
nuisance vector. `render_state_id` is derived from `science_state_id`,
`nuisance_state_id`, and `render_system_contract_hash`.

Create the compact production-plan transfer package after materialization:

```bash
python3 work/experiments/ml/datasets/materialize_master_v4.py \
  --outdir work/experiments/ml/datasets/materialized/master_v4 \
  --package-only
```

The package contains the frozen V4 state-plan and handoff inputs needed by the
cluster renderer: master prescription, vector spaces, unit contract, joint
base envelope, nuisance bank, render-system contract, render contract, subset
registry, freeze manifest, review decisions, state-plan JSONLs, QA summaries, and
`RENDER_HANDOFF.md`. It excludes tiny development materialization, notebook
caches, local audit scratch, historical dataset context, and rendered FITS
images. Verify the package after transfer with `shasum -a 256 <package>` on
macOS or `sha256sum <package>` on Linux.

Accepted V4 QA findings are recorded in `qa/review_decisions.json`. The
`joint_full_v4` population uses four balanced multiscale strata and the
accepted anisotropic Fisher envelope. M1 contributes a relatively large share
of aggregate squared Fisher radius because it has eight coordinates and a wider
accepted envelope than M2; no single coordinate is marked as an unintended
dominant driver. `radial_capture_v4` directions are isotropic proposals
conditioned on the fixed V4 sampling/feasibility envelope. High-radius bins,
especially `1000-1500` and `1500-2000`, are therefore feasible-direction
conditioned populations, not unconditioned isotropic shells.
`boundary_stress_v4` is deferred for the first render campaign.
