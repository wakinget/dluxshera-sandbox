# S07 V4 Joint-Full Bring-Up

S07 introduces the prepared V4 joint-full population with family A pairs,
`same_pair_id: false`, all nuisance states, no noise, and the shared
train-derived V4 global-max-abs scaler.

The matrix expands to ten runs. Validation stays on the frozen full V4
validation population and `evaluate_test: false` for every run.
