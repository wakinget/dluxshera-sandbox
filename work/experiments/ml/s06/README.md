# S06 V3 Large-Model Optimizer Bridge

S06 is a V3-only nine-run bridge from the S01/S05 frozen benchmark contract to
the larger `concat_diff` architecture and the longer fixed-LR training budget.

All runs keep `evaluate_test: false`. The frozen S01 test pair manifest is
identity-checked for contract completeness but is not evaluated during model
selection.
