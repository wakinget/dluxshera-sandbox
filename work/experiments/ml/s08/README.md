# S08 V4 Pair-Family and Nuisance Study

S08 evaluates V4 pair families C and A/B/C mixtures with the large S05-style
architecture. `S08-E03` enables multitask nuisance-delta prediction while
checkpoint selection remains science validation loss. `S08-E04` uses a
separate nuisance-holdout split for training and reserves nuisance states 8-9
for post-training audit only.

All twelve production runs keep `evaluate_test: false`.
