# S10 V4 Fisher Eigenmode-Aware Training Loss

S10 uses the canonical S07 large `concat_diff` Siamese CNN with the V4
`joint_full_v4` family-A pair policy, grouped split registry, and global-max-abs
scaler. S10-E01 is the shared clean ordinary-loss reference cohort (3 seeds)
reused as-is by S11 and S12. S10-E02 and S10-E03 reweight the science loss
toward strong- and weak-curvature Fisher eigenmodes respectively, using the
frozen `S10-V4-SCIENCE-FIM-EIGENBASIS-v1` eigenbasis artifact.

All runs keep `evaluate_test: false`. See
`docs/dev/notes/ml_s10_s12_campaign_plan.md` for the full launch prescription;
as of that note this is repository preparation, not evidence that jobs have
been submitted or completed.
