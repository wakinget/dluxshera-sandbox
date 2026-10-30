# S06 support and provenance

The authoritative scientific interpretation is [review.md](review.md).
The [study prescription](../../s06/README.md) defines the experiment;
the [ML study index](../README.md) tracks review status.

## Storage and reproduction

All following Results paths are relative to the repository root and ignored by git.
The input archive `Results/ml_analysis/S06/_review_import_20261030/` is retained
unchanged as noncanonical historical input, including its original script.
Its script duplicates `campaign_analysis.py` initializer, parameter, slice,
distance-bin, and geometry calculations and `campaign_plots.py` figures; it also
hardcodes `/mnt/data`. It is not a maintained entry point. Use the existing
`load_campaign([Path("Results/hpc_imports/ml/ls6/S06")])` and
`campaign.load_predictions(run_id)` for validation-only saved-array analysis.
No new parallel analysis framework is introduced.

Support was copied byte-for-byte from that input into:

- `Results/ml_analysis/S06/tables/`: all eight supplied derived CSVs.
- `Results/ml_analysis/S06/figures/`: all seven supplied plots.
- `Results/ml_analysis/S06/source_eigenmodes/`: explicitly an **analysis copy**,
  not the authoritative sweep. Includes the full 20D `eigenvectors_long.csv`
  (all 400 components), FIM, labels, eigenvalues, manifest and SHA256SUMS.json.

The original remains authoritative at
`Results/ml_eigenmodes/ml_dataset_v3_flight_simple_50ms_psf160/sweep_schur_marginalized/`.
These five source files were verified against the analysis copy and hashes below.
Other original sweep variants exist locally but are outside this review.
The reference is V3 flight-simple, 50 ms, PSF160, Schur registration-nuisance
marginalized with pseudoinverse rcond=1e-12. Operating points/reference datasets
can differ from S06 validation; the dirty source tree recorded by the runs is
not recoverable from the compact manifests alone.

S06 imported run artifacts live at `Results/hpc_imports/ml/ls6/S06/`.
The handoff identifies `s06_s09_review_20261008.tar.gz`, `runs/S06/`, as its
validation source. Generated tables are frozen handoff products, not newly
regenerated outputs from the maintained loader. Their schema differs from the
loader's tables. Recomputing residual covariance PCA requires saved scaled-z
arrays; it is separate from diagonalizing the physical Schur FIM. No residual
projection into that FIM basis is supported until transform and model identity
are validated.

## Weak physical modes

From `tables/schur_eigenmode_summary.csv`; ranks descend across all 20 modes,
indices are zero-based. Coefficients are unit-Euclidean physical packed-basis
loadings; signs are arbitrary. They are asymmetric and cross-coupled, with
mixed parameter units. All eigenvalues are positive. Eight numerical weak flags
use 1e-12 times the largest eigenvalue, not an exact null-space claim.

| Rank (index) | Eigenvalue | Five prominent signed loadings |
| --- | ---: | --- |
| 13 (12) | 0.027846 | M1 Z4:+0.4976; M2 Z7:-0.4472; M2 Z8:-0.4065; M2 Z4:-0.3757; M2 Z11:+0.3083 |
| 14 (13) | 0.026285 | M2 Z7:-0.5637; M1 Z4:-0.4242; M2 Z5:-0.3125; M2 Z4:+0.3100; M2 Z11:-0.2668 |
| 15 (14) | 0.024141 | M2 Z8:+0.6202; M2 Z5:-0.5070; M2 Z6:+0.3546; M1 Z5:+0.2731; M1 Z8:-0.2431 |
| 16 (15) | 0.022317 | M2 Z7:+0.5604; M2 Z6:+0.4643; M2 Z8:-0.4613; M1 Z6:-0.2496; M2 Z5:-0.2266 |
| 17 (16) | 0.021507 | M2 Z5:+0.5652; M2 Z6:+0.5430; M1 Z5:-0.3026; M1 Z6:-0.2905; M2 Z9:+0.2570 |
| 18 (17) | 0.019172 | M2 Z9:+0.8453; M1 Z9:-0.3315; M2 Z5:-0.2191; M2 Z8:-0.1568; M1 Z4:-0.1288 |
| 19 (18) | 0.015698 | M2 Z10:-0.8802; M1 Z10:+0.3447; M2 Z6:+0.1973; M2 Z9:-0.1384; M2 Z8:-0.1148 |
| 20 (19) | 0.003871 | M2 Z4:-0.7693; M2 Z11:-0.5941; M1 Z11:+0.1701; M1 Z4:-0.1608; M2 Z8:+0.0124 |

Full vectors: `Results/ml_analysis/S06/source_eigenmodes/eigenvectors_long.csv`;
weak-only table: `Results/ml_analysis/S06/tables/schur_weak_eigenvectors_physical_basis.csv`.
Plots: `Results/ml_analysis/S06/figures/schur_eigenvalue_spectrum.png`,
`schur_weak_eigenvectors_heatmap.png`, and
`matched_mirror_correlation_comparison.png` (same figures directory).
FIM sensitivity correlations, learned residual correlations and centered
residual-covariance PCA measure different objects. Z4–Z10 matched FIM
correlations are 0.99991–0.99997; Z11 is about +0.280.

## Source SHA256

| File | SHA256 |
| --- | --- |
| manifest.json | `882dbe42e5e61f3f37ec259183eba7e738bc91a1976107950457e89d584d8c07` |
| eigenvalues.csv | `71646321de02a241840f40d21c2beb87e0167497362143ce3ff8002a58be3952` |
| eigenvectors_long.csv | `d47ea2742538b7771cd0d59a0153b01e78c17bd600b498ca8eb2f06d883e3a30` |
| fim_matrix.csv | `380202cb607c7271db522a2aaaf7948df82f8292100ef8a51c7523c583eeb65b` |
| parameter_labels.csv | `c79f6c499842a24ef06763cf1137e61db360a4cd5052af99291f09fb88d75b10` |

## Reconciled archive versus mirror identity

The [reconciliation](reconciliation.md) establishes the earlier-generation
work mirror as historical, preserved outside active discovery. The production
archive was found locally, hashed and checked directly; its nine saved-array
RMSEs reproduce the review. `Results/hpc_imports/ml/ls6/S06/` now contains that
verified production population with archive provenance and per-file hashes.
See the reconciliation for the source comparison, exact paths and remaining
remote-origin and dirty-tree limitations.
