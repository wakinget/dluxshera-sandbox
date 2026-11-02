# S10 — Fisher eigenmode loss review

Review date: 2026-11-02. Preliminary closeout: principal diagnostics reproduced;
parameter, distance, weight-distribution and history interpretation remain pending.

## Motivation and controlled design

S10 asks whether Fisher eigenmode weighting yields a more useful optical-estimator
initializer. E01 uses ordinary science loss, E02 strong-mode weighting (0.5), and
E03 weak-mode weighting (0.5). The larger concat_diff CNN and V4 joint_full_v4
family-A prescription are shared, with seeds 11, 23 and 47.

## Provenance and evaluation contract

The installed nine-run collection is complete through epochs 0–599; see
[reconciliation](reconciliation.md). Budget completion does not establish
optimization convergence. Saved predictions represent selected best checkpoints.
All manifests record test_evaluated=false. Source revision
`0f20f63ada40677d12068bfc89d90d6be1bdd942` has dirty=true: the commit alone cannot
reproduce the executed tree. S11 has a different frozen validation identity and
is not a paired ordinary-loss comparison.

## Quantitative findings

Arithmetic means of three seed RMSEs, not pooled RMSE:

| Experiment | Global Fisher RMSE | Separation RMSE (mas) | Strong 1–6 | Middle 7–14 | Weak 15–20 |
|---|---:|---:|---:|---:|---:|
| E01 | 122.7231 | 3.8121 | 17.65 | 20.25 | 222.15 |
| E02 | 163.1933 | 2.0020 | 6.59 | 8.63 | 297.71 |
| E03 | 125.197 | 4.569 | 28.07 | 25.73 | 224.90 |
| Post-hoc hybrid | 121.8456 | 1.9983 | — | — | — |

E02 improves separation by 47.5% while worsening global error by 33%; both
patterns hold in all three seeds. E03/E01 weak-group RMSE ratios are 1.0076,
1.0277 and 1.0023 for seeds 11, 23 and 47: no seed improves this group.
For MSE deltas, negative means improvement; positive means regression.

## Eigenmode geometry and hybrid

Residuals in Fisher-scaled coordinates are rotated as `(prediction-truth) @ Q`,
where Q has orthonormal eigenvectors as columns. Rotation preserves summed
squared error. This is a fixed 20-dimensional science-block basis, holding
registration nuisance parameters fixed, not a nuisance-marginalized Schur basis.
Physical separation is obtained with its Fisher scale and converted to mas.

Separation squared overlaps are 0.5283 (mode 14), 0.3530 (2), 0.0635 (8),
and 0.0412 (13). Modes 14 and 2 contain 88.1% of that coordinate direction.
Overlap alone does not determine error variance: amplitudes and cross-mode
covariance matter. E02's weak-subspace separation residual is about 0.230 mas
RMS in every seed despite tiny overlap. Mode 14's improvement should not be
read as a proportional response to its explicit normalized weight (~0.373).

The hybrid uses E02 predictions in modes 1–14 and E01 in 15–20, after exact
pair_record_id, evaluation-slice and truth-array equality checks. Its split was
chosen after inspecting validation results. It is neither a trained standalone
estimator nor independently validated evidence of generalization.

## Interpretation and reproducibility

The weak six modes dominate the global squared-error budget. Global parameter
RMSE can therefore rank initializers differently from separation RMSE. A useful
CNN initializer need not minimize global RMSE if it improves downstream capture
or final recovery; these data have not measured either. Nonlinear optical
coupling could make weak-mode errors harmful to an iterative estimator.
No convergence basin, microarcsecond precision or full instrumental-error-budget
claim follows from these milliarcsecond validation errors and three seeds.

Run `python work/experiments/ml/analysis/s10_s11_diagnostics.py` from repository
root. It reads compact validation arrays only, prints all 20 mode RMSEs and
checks rotation invariance and stored global metrics. Redirect output into
ignored `Results/ml_analysis/S10/closeout/`. Remaining closeout work includes
full weight reconstruction, parameter and distance tables, focused tests and
strict documentation validation.

## Follow-up questions (not approved experiments)

1. Compare E01/E02/E03 and hybrid initializations on common synthetic observations
   with the iterative estimator: success, iterations, separation bias/error,
   image residual and weak-mode sensitivity, including controlled perturbations
   and regularized/truncated alternatives.
2. Compare direct separation weighting with participation-weighted eigenmodes,
   retaining global and weak-mode guardrails.
3. Develop a practical hybrid without validation-selection leakage.
4. Evaluate S11 constraints under common validation and downstream recovery.
5. Revisit noise and broader nuisance robustness after S12 and status audits.
