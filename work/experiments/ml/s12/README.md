# S12 V4 Photon-Noise Observation Robustness

S12 reuses the S10-E01 clean reference cohort (`S10-E01-R001..R003`) rather
than retraining a separate baseline. S12-E01 trains with dynamic SHERA photon
observation noise and evaluates against a fixed-seed noisy validation
manifest. S12-E02 adds the same dynamic noise plus a modest (weight 0.05)
independent-realization prediction-consistency loss.

All runs keep `evaluate_test: false`. See
`docs/dev/notes/ml_s10_s12_campaign_plan.md` for the full launch prescription;
as of that note this is repository preparation, not evidence that jobs have
been submitted or completed.
