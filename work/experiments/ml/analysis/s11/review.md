# S11 — Pairwise consistency review

Review date: 2026-11-02. Preliminary closeout: principal diagnostics reproduced;
full history analysis and validation checks remain pending.

## Motivation and design

E01 adds antisymmetry weight 0.1 to ordinary science loss. E02 additionally
uses identity weight 0.02. Matched seeds are 11, 23 and 47. This tests the
incremental identity constraint, with the larger concat_diff CNN prescription.

## Provenance and comparability

All six installed runs have histories for epochs 0–599 and test_evaluated=false.
The 31 installed files match normalized staging byte-for-byte, and export
inventory hashes match. Archive SHA256 is
`ef0cb6346cc0e48e9bc52c1bc9aeec8a28f19a7fdc93b8bb0d94994fe7ad6b1e`.
Source `0f20f63ada40677d12068bfc89d90d6be1bdd942` is dirty, so the commit does not
fully identify executed source. S10-E01 is the prescribed reference, but its
frozen-validation artifact differs; cross-study aggregates are not paired.

## Findings

| Experiment | Global Fisher RMSE | Separation RMSE (mas) | Mean antisymmetry MSE |
|---|---:|---:|---:|
| E01 | 122.521 | 3.861 | 53.0656 |
| E02 | 122.408 | 4.154 | 53.6072 |

| Seed | E01 global RMSE | E02 global RMSE | E01 antisymmetry MSE | E02 antisymmetry MSE | E02 identity MSE |
|---|---:|---:|---:|---:|---:|
| 11 | 122.604 | 119.725 | 62.7696 | 34.3545 | 2.9087 |
| 23 | 124.975 | 125.236 | 55.6142 | 31.7894 | 2.1958 |
| 47 | 119.984 | 122.262 | 40.8131 | 94.6775 | 15.0907 |

The approximately 0.09% global improvement has mixed seed behavior; separation
worsens 7.6%. Identity regularization does not consistently improve ordinary
regression or antisymmetry. Seed 47 is particularly unfavorable.

Antisymmetry is a distinct consistency residual, not ordinary prediction error.
E01 identity is **not evaluated**: its serialized zero is a disabled-term
sentinel, not perfect identity consistency. E02 identity values cannot establish
improvement against that unavailable baseline. Three seeds provide limited
precision; budget completion is not evidence of optimization convergence.

## Reproducibility and limitations

Run `python work/experiments/ml/analysis/s10_s11_diagnostics.py` from the repository
root. Ordinary global RMSE is recomputed from saved arrays; consistency values
are reported from stored metrics, not independently inferred from forward-only
predictions. No checkpoints, model inference or locked tests are used.

The results do not establish antisymmetry benefit relative to S10-E01. That
requires common frozen-validation evaluation, potentially authorized checkpoint
inference. Nor do they establish downstream iterative convergence, capture
radius, final microarcsecond precision or instrumental-budget robustness.

## Follow-up

Prioritize CNN-to-iterative-estimator convergence, then separation-targeted loss,
practical hybrid initialization, common-protocol consistency comparisons and
noise/nuisance robustness. These are candidate investigations only; see the
[detailed S10 handoff](../s10/review.md#follow-up-questions-not-approved-experiments).
