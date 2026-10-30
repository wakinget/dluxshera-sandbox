# S07 V4 Joint-Full Bring-Up

S07 introduces the prepared V4 joint-full population with family A pairs,
`same_pair_id: false`, all nuisance states, no noise, and the shared
train-derived V4 global-max-abs scaler.

The matrix expands to ten runs. Validation stays on the frozen full V4
validation population and `evaluate_test: false` for every run.

`joint_train_prefix_size` in E03–E06 restricts the eligible **training science
states** of `joint_full_v4` to a nested prefix of the frozen 65,536-state train
plan; each state is still rendered against all ten fixed nuisance states, so a
prefix record is not an independent physical configuration.

The campaign is complete and reviewed: see the
[S07 scientific review](../analysis/s07/review.md) and its
[provenance reconciliation](../analysis/s07/reconciliation.md).
