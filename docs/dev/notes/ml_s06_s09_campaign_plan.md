# S06-S09 ML Campaign Plan

Date: 2026-09-09

This note records the planned S06-S09 bring-up. It is a launch preparation
plan, not evidence that V4 prepared artifacts or cluster runs have been
materialized.

## Scope

- S06: V3-only bridge on the frozen S01/S05 prepared-data, split, validation,
  and test-manifest contract. Nine runs.
- S07: V4 joint-full family A bring-up with baseline and large models plus
  nested joint-train prefixes. Ten runs.
- S08: V4 pair-family/nuisance study, including S08-E03 multitask science plus
  nuisance-delta training and S08-E04 unseen-nuisance audit. Twelve runs.
- S09: V4 broad-joint plus distance-controlled-radial study with family A
  pairs and radial-only easy-to-hard distance curriculum. Nine runs.

The tracked prescriptions live in:

- `work/experiments/ml/s06/study.yaml`
- `work/experiments/ml/s07/study.yaml`
- `work/experiments/ml/s08/study.yaml`
- `work/experiments/ml/s09/study.yaml`

The repository-side expansion/audit command is:

```bash
PYTHONPATH=src python3 work/experiments/ml/materialize_study_artifacts.py audit-study \
  --study work/experiments/ml/s06/study.yaml \
  --study work/experiments/ml/s07/study.yaml \
  --study work/experiments/ml/s08/study.yaml \
  --study work/experiments/ml/s09/study.yaml
```

Expected output: S06 = 9, S07 = 10, S08 = 12, S09 = 9, total = 40, and
`all_evaluate_test_false: true`.

## Artifact Discipline

V4 production studies require a portable artifact lock. The lock is the
authoritative identity bundle for:

- prepared dataset identity;
- split registry identity;
- shared V4 scaler identity;
- primary validation pair manifest;
- test manifest when materialized;
- required audit manifests when declared.

Physical filesystem roots are excluded from scientific prepared-data identity.
Copying the same prepared artifact from Gattaca2 to Lonestar6 should preserve
the identity.

S08 has two artifact profiles:

- `standard`: used by S08-E01, S08-E02, and S08-E03. It uses
  `SPLIT-V4-ROLE-PRESERVING-v1`, all ten nuisance states in the train/dev
  nuisance partition, `SCALER-V4-GLOBAL-MAX-ABS-v1`, and
  `S08-STANDARD-ARTIFACT-LOCK-v1`.
- `nuisance_holdout`: used by S08-E04. It uses
  `SPLIT-V4-S08-NUISANCE-HOLDOUT-v1`, with nuisance bank entries 0-7 assigned
  to the seen/train partition and entries 8-9 assigned to
  `unseen_nuisance`, plus `S08-HOLDOUT-ARTIFACT-LOCK-v1`.

S08 primary validation is experiment-specific. S08-E01 selects
`validation_c`; S08-E02 and S08-E03 select `validation_abc`; S08-E04 selects
`validation_abc_seen` and the holdout-profile `test_seen` manifest. The
S08-E04 `unseen_nuisance_audit` is a post-training audit artifact and must
never control checkpoint selection.

S09 is not a globally distance-balanced study. Its source-family sampling is
conditional:

- S09-E01: 75% broad `joint_full_v4` A-pair sampling plus 25%
  distance-balanced `radial_capture_v4` A-pair sampling.
- S09-E02: 50% broad `joint_full_v4` plus 50% distance-balanced
  `radial_capture_v4`.
- S09-E03: 50/50 broad joint/radial mixture, with the easy-to-hard curriculum
  applied only to the radial family. The joint family remains broad at every
  epoch.

The balanced radial bins are `0-100`, `100-250`, `250-500`, `500-1000`,
`1000-2000`, and `2000-5000`. In S09-E03 the radial curriculum uses bins
through 500 for epochs 0-99, through 2000 for epochs 100-249, and through 5000
for epochs 250 and later.

## Training Objective

S08-E03 retains the normal science-correction head and adds a nuisance-delta
head for registration x, registration y, and roll. The total loss is:

```text
loss_total = loss_science + lambda_nuisance * loss_nuisance
```

The default `lambda_nuisance` is 1.0 after component-wise nuisance scaling.
Checkpoint selection remains based on science validation loss for this first
experiment.
