# SHERA ML Master Dataset V4 Specification

Status: frozen scientific/render contract for the completed first Gattaca2 V4
production render. This document records state-plan and render identity; it
does not authorize additional rendering, transfer, or cluster submission.
Execution completion evidence is recorded in `README.md` and
`docs/dev/notes/ml_program_status_20260908.md`.

Frozen local contract root:
`work/experiments/ml/datasets/materialized/master_v4`

Frozen master scientific content hash:
`dcc46d19cd87a65c2321b2586543c30478d1917b597dd8097397adcebea43fe4`

## Reviewed Decisions

- The first V4 campaign changes science-state geometry while holding the
  learned S01/S05 coordinate system and forward-problem contract fixed.
- The authoritative ordered science vector and Fisher diagonal scales are from
  `Results/ML Training Datasets/preprocessed/PREP-V3-nuisance-v1/vector_spaces.json`
  with SHA256
  `8bb1b9ed7cae505d743b4f1fc169ea47d695d9d8dc1aa37b85f1e30488a8ab0f`.
  Specifically, V4 uses the stored order of
  `spaces.fisher_scaled_delta.components` and the matching stored
  `transforms.fisher_diagonal_scale.scales`.
- The matching authoritative raw parameter contract is
  `Results/ML Training Datasets/shera_training_dataset_nuisance_pairs_20260511/parameter_space.json`
  with SHA256
  `b441a568268bf8b1d91923e1a75501595467e64b22714a0ac665c7b1fbff79ff`.
  Raw records are reconciled by component label into the prepared V3 order
  before deriving nominal values, historical envelopes, physical vectors,
  Fisher-scaled vectors, normalized summaries, compatibility rows, or IDs.
  Duplicate, missing, or unexpected raw labels fail materialization.
- The sparse-100k parameter contract was compared. Exact JSON float identity
  differs for several components at representation precision, so V4 uses the
  S01/S05 nuisance-pair contract. The component comparison is materialized at
  `compatibility/coordinate_compatibility_comparison.json`.
- A current-branch Fisher recomputation was not performed. The stored
  training-coordinate scales were preserved; recomputation remains diagnostic
  only.
- The selected V4 nuisance bank is the recovered S01/S05 V3 nuisance-pair
  10-state bank, identity `nuisance_bank_v4_s01_s05_v3_training_10`, hash
  `1eb0ffe3058f4416ac0cc732414956cc9ccb55ea126c5405544003ef5d288a56`.
  Nuisance vectors use the explicit canonical order
  `source.x_position_as`, `source.y_position_as`,
  `source.position_angle_deg`, verified against the selected V3 nuisance
  manifest and written by label from row mappings.
- The sparse-100k nuisance bank is retained separately as
  `nuisance_bank_legacy_sparse100k_10`. It differs from the S01/S05 bank for
  all 10 indexed nuisance states. The comparison is materialized at
  `compatibility/nuisance_bank_comparison.json`.
- Historical V3 sweep extrema are sampling envelopes, not physical validity
  constraints. No V4 artifact treats them as model-validity bounds.
- `qa/review_decisions.json` records accepted joint geometry, accepted radial
  feasibility conditioning, and `boundary_stress_v4` deferred for the first V4
  render campaign. Review-decision hash:
  `cedfacc707cf477e501556dc18eac2864070325fcb45ba30097601cb72d5ac85`.

## Vector And Unit Contract

The frozen aggregate V4 vector-space contract hash is
`732770f26d0c572315c56816f5d0ad6583d317ab346d8e92b54d712c52d2b165`.
The science vector-space ID is
`916a005234f944569f9e635873da6b79a6df505e26fcfd002ffb389cdec19f1b`.
The nuisance vector-space ID is
`abe2f0435286941c60ccadb4a0601fb2ff4dca5b470c5aeb31fec6f77357a85c`.
The machine-readable contract is `vector_spaces.json`; `unit_contract.csv`
contains the same unit table in row form.

| component class | unit |
| --- | --- |
| `source.separation_as` | arcsec |
| `source.log_flux_total` | `log10(detected_photons)` |
| `source.contrast` | dimensionless |
| `optics.plate_scale_as_per_pix` | arcsec / pixel |
| `optics.primary.zernike_coeffs_nm[*]` | nm |
| `optics.secondary.zernike_coeffs_nm[*]` | nm |
| `source.x_position_as` nuisance | arcsec |
| `source.y_position_as` nuisance | arcsec |
| `source.position_angle_deg` nuisance | deg |
| Fisher-scaled coordinates | dimensionless |

`source.log_flux_total` follows the repository source-photometry convention:
`log10` of total detected photons from both source components over the modeled
exposure after collecting area and throughput.

## Science Families

Primary first-campaign V4 science-state counts are frozen as:

| family | train | validation | test | total |
| --- | ---: | ---: | ---: | ---: |
| `joint_full_v4` | 65,536 | 8,192 | 8,192 | 81,920 |
| `radial_capture_v4` | 16,384 | 4,096 | 4,096 | 24,576 |
| total | 81,920 | 12,288 | 12,288 | 106,496 |

Every V4 science state is rendered against every selected nuisance state by
compact index contract, not by writing duplicate-expanded plan rows.

## Joint Full V4

The frozen joint base Fisher envelope is a sampling envelope:

```text
joint_base_halfwidth_i = min(historical_max_sigma_i, 1000)
```

This preserves tighter historical envelopes, keeps contrast at 200 sigma and
secondary Zernikes at 500 sigma, and caps the exceptional plate-scale historical
20,000-sigma sweep at 1,000 sigma for the full-dimensional joint family.

Scale strata are equally represented:

| scale multiplier | train | validation | test |
| ---: | ---: | ---: | ---: |
| 0.125 | 16,384 | 2,048 | 2,048 |
| 0.25 | 16,384 | 2,048 | 2,048 |
| 0.5 | 16,384 | 2,048 | 2,048 |
| 1.0 | 16,384 | 2,048 | 2,048 |

Independent scrambled Sobol sequences are used for each
`split x scale_stratum`. Seeds are SHA256-derived from stable tokens:
dataset version, family, split, and scale stratum.

Training rows are interleaved in ascending stable scale-stratum order. The
standard nested prefixes preserve an exact 1:1:1:1 mixture:

| train prefix | states per stratum |
| ---: | ---: |
| 4,096 | 1,024 |
| 8,192 | 2,048 |
| 16,384 | 4,096 |
| 32,768 | 8,192 |
| 65,536 | 16,384 |

Train Fisher-radius percentiles by scale stratum:

| scale | p01 | p50 | p99 | max |
| ---: | ---: | ---: | ---: | ---: |
| 0.125 | 184.096 | 259.753 | 325.813 | 362.516 |
| 0.25 | 368.721 | 519.559 | 651.835 | 737.899 |
| 0.5 | 733.228 | 1038.598 | 1305.950 | 1491.593 |
| 1.0 | 1462.387 | 2075.038 | 2603.314 | 2983.627 |

Combined joint-full radius percentiles are p01 203.514, p50 678.438, p95
2277.598, p99 2486.501, max 2983.627.

Squared Fisher-radius contribution QA:

| family | fraction of `||z||^2` |
| --- | ---: |
| source | 0.156442 |
| plate scale | 0.076687 |
| M1 Zernike | 0.613497 |
| M2 Zernike | 0.153375 |

The plate-scale fraction is 0.076687 and contrast fraction is 0.003067. The
only domination flag is the M1 Zernike family share, expected from eight
full-width primary-Zernike dimensions; no single coordinate dominates.

Frozen joint plan hashes:

| split | SHA256 |
| --- | --- |
| train | `1a0c63e86e16d1f3fee7d386f23341294f36e4063338093bac180f2c4d81afa4` |
| validation | `65cee283ef3d16be1beefc3b0e5f1c02ea07b691570a2a8ff37fa3d9416855b0` |
| test | `5c43d6190c06c5834e4f15257f3c8ca884439dc1b5d5c9d55e4997513fd98fa4` |

Joint QA hash:
`7adebfd7e011b7b8e54f8b73f2ba570e6cfc3fbf647abf934d3651689cf9b0d6`.

## Radial Capture V4

Radial capture uses deterministic full-dimensional directions generated
independently from the requested radius. For each proposal, the planner computes
`r_feasible` against the V4 sampling envelope and accepts only if:

```text
requested_radius <= r_feasible
```

Component clipping and radius shrinking are not used. Accepted rows record
requested radius, actual radius, feasible radius, attempt count, direction
sequence index, and radial bin. Normally actual radius equals requested radius.

Frozen radial bins:

| bin | train | validation | test |
| --- | ---: | ---: | ---: |
| 0-100 | 2,731 | 683 | 683 |
| 100-250 | 2,731 | 683 | 683 |
| 250-500 | 2,731 | 683 | 683 |
| 500-1000 | 2,731 | 683 | 683 |
| 1000-1500 | 2,730 | 682 | 682 |
| 1500-2000 | 2,730 | 682 | 682 |

Integer remainders are assigned deterministically to lower-index bins.

Acceptance rates by split/bin:

| split | 0-100 | 100-250 | 250-500 | 500-1000 | 1000-1500 | 1500-2000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1.000 | 1.000 | 0.975 | 0.743 | 0.259 | 0.033 |
| validation | 1.000 | 1.000 | 0.972 | 0.751 | 0.275 | 0.034 |
| test | 1.000 | 1.000 | 0.973 | 0.718 | 0.275 | 0.031 |

Direction-conditioning QA flags the contrast coordinate in the 1000-1500 and
1500-2000 bins, with direction standard-deviation ratios of 0.437 and 0.312
relative to the lowest bin. This is recorded for review and was not silently
modified.

The accepted interpretation is that radial directions are isotropic proposals
conditioned on the fixed V4 sampling/feasibility envelope. High-radius
`1000-1500` and `1500-2000` states are feasible-direction-conditioned
populations, not unconditioned isotropic shells.

Frozen radial plan hashes:

| split | SHA256 |
| --- | --- |
| train | `b2d39621fd22cf93f9524ccf953c6c74783c1a17a7867d5a8cd69a840baec69d` |
| validation | `a7e7e21ba07e9eae089882ae15444a43898f5a847803f9660892732c244f01de` |
| test | `5b21365da25b9cabe735c8972244583f92ed5c48acd4cb96448a5aee65cc5639` |

Radial QA hash:
`07fb03c2116a7e9595af223600d0209318c5f4509d9ca236c3f8be3e05285f19`.

High-radius states above 2000 Fisher radius remain scientifically interesting
but are deferred to a future reviewed family such as
`high_radius_feasible_v4` or `boundary_stress_v4`.

## Stable Identity Semantics

`science_state_id` hashes canonical scientific state content:

- dataset version;
- family;
- split role;
- science vector-space identity, independent from nuisance identity;
- ordered absolute physical science vector;
- ordered Fisher delta vector;
- sampling-family contract identity.

`nuisance_state_id` hashes the nuisance vector-space identity, independent
from science identity, and the ordered physical nuisance vector.

`render_state_id` hashes `science_state_id`, `nuisance_state_id`, and the
render-model/system contract identity.

Filesystem path, timestamp, SLURM id, and shard number do not participate in
scientific or render identity.

The render-system contract hash is
`26fdb799813affefad51585c074f24f32841501036007992b9dfdebdfeb1deab`.
It is generated from the authoritative resolved S01/S05-compatible `system`
subtree plus noise policy. Local file paths are provenance; referenced files
contribute by content hash when present.

The split-integrity summary hash is
`8cfeab02343f3e37cc537c1e371282074aef10a09cff018386e9c4c5c67d652c`.
It validates science IDs and canonical ordered physical science vectors across
family and split boundaries.

## Compact Render Index

The render contract hash is
`566d433b3510fccc6d91b983b9d03a20469e2400039e24987c0e1714b7b7ba8c`.

The compact bijection is:

```text
render_index = science_global_index * nuisance_count + nuisance_bank_index
science_global_index = render_index // nuisance_count
nuisance_bank_index = render_index % nuisance_count
```

Frozen counts:

```text
106,496 science states x 10 nuisance states = 1,064,960 renders
```

The render contract preserves family, split, global science index, plan path,
plan hash, nuisance bank index, nuisance vector, and render identity inputs. It
does not assume one flat output directory or any particular FITS/metadata shard
layout.

## Storage Estimate And Completed Footprint

`render_scale_summary.json` hash:
`d7ffab87e9121d6c580ec4d2c5e2695072de6dccb1e5afc040cf859b88a683db`.

The pre-render planning estimate for 1,064,960 renders at 160 x 160 was:

| item | bytes |
| --- | ---: |
| raw float64 payload | 218,103,808,000 |
| expected FITS payload | 223,897,190,400 |
| expected metadata payload | 4,362,076,160 |
| potential prepared float32 payload | 109,051,904,000 |
| total estimated persistent footprint | 337,311,170,560 |

The completed first production render measured the canonical raw corpus at
123 GB under:

```text
/projects/shera_hpc/data/ml_training/shera_ml_master_v4
```

The filesystem audit found 1,064,960 FITS files and 1,064,960 JSON sidecars
with matching totals. After completion, `/projects/shera_hpc` reported 3.2 TB
size, 878 GB used, 2.3 TB available, and 28% utilization. The measured 123 GB
corpus size supersedes the earlier conservative planning estimate for the raw
FITS + JSON corpus. The potential prepared float32 payload remains a future
derived-data planning value, not an existing V4 prepared dataset.

For future render launches, explicit repair tasks, or capacity audits, rerun:

```bash
df -h /projects/shera_hpc
du -sh /projects/shera_hpc/data/ml_training/*
```

## Frozen Artifacts

Required final V4 state-plan artifacts were materialized locally:

- `master_v4.yaml`
- `freeze_manifest.json`
- `vector_spaces.json`
- `unit_contract.csv`
- `joint_base_envelope.json`
- `nuisance_bank.json`
- `render_system_contract.json`
- `compatibility/coordinate_compatibility_comparison.json`
- `compatibility/nuisance_bank_comparison.json`
- `state_plans/joint_full_v4/{train,validation,test}.jsonl`
- `state_plans/radial_capture_v4/{train,validation,test}.jsonl`
- `qa/split_integrity_summary.json`
- `qa/review_decisions.json`
- `qa/joint_geometry_summary.json`
- `qa/joint_squared_radius_contribution.csv`
- `qa/radial_feasibility_summary.json`
- `qa/radial_direction_conditioning.csv`
- `production_qa.ipynb`
- `render_contract.json`
- `render_scale_summary.json`
- `subset_registry.json`
- `RENDER_HANDOFF.md`

Generated materializations, QA outputs, freeze manifests, render contracts, and
transfer packages are not tracked in Git. The compact transfer package for
cluster handoff is generated from the ignored materialization root and verified
by SHA256. Current local package:
`shera_ml_master_v4_dcc46d19cd87a65c_stateplans.tar`, SHA256
`34d0d907696dd6b84c691152ceead2c1cfe6caae3ca2a4ae010d826014add078`.

Subset-registry hash:
`4f539b4ea46b387314b9f7dcafe3cf17712c1fc630dfc68b04399f403098fcd0`.

No unresolved scientific decision remains in this specification that would
change the first-campaign production science state vectors.
