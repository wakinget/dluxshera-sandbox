# Presets directory migration note

Preset files are now split by block scope:

- System presets: `src/dluxshera/data/system_presets/`
- Experiment presets: `src/dluxshera/data/experiment_presets/`

Use `system.preset` to choose a system preset and `experiment.preset` to choose an
experiment preset. System preset files must only contain `system`, and experiment
preset files must only contain `experiment`.
