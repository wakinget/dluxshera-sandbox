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
image corpus unless future versions explicitly add that option.

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

## Next Task

The next task should implement deterministic V4 state-plan materialization and
QA from `master_v4_spec.md`: ordered vector-space metadata, nested Sobol
`joint_full` plans, controlled-radius `radial_capture` plans, nuisance-bank
materialization, cross-product expansion, pre-render QA, and immutable state
plan manifests.
