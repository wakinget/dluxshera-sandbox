# S11 V4 Pairwise Physics Consistency

S11 reuses the S10-E01 clean reference cohort (`S10-E01-R001..R003`) rather
than retraining a separate baseline. S11-E01 adds an explicit reverse-pair
antisymmetry constraint (weight 0.1) on top of the ordinary supervised science
loss. S11-E02 adds a low-weight identity constraint (weight 0.02) requiring
f(A,A) = 0 and f(B,B) = 0.

All runs keep `evaluate_test: false`. See
`docs/dev/notes/ml_s10_s12_campaign_plan.md` for the full launch prescription;
as of that note this is repository preparation, not evidence that jobs have
been submitted or completed.
