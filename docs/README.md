# Documentation map

Quick pointers to the main documentation areas without duplicating content.

## Architecture
Design and decisions for the system’s core components and data flow.
- [Systems and binders](architecture/systems_and_binders.md)
- [Detector layering and contracts](architecture/systems_and_binders.md#detector-layers-and-detector-builder)
- [Parameters and store](architecture/params_and_store.md)
- [Inference and loss](architecture/inference_and_loss.md)
- [Testing approach](architecture/testing.md)
- ADRs: [0001 core architecture foundations](architecture/adr/0001-core-architecture-foundations.md) · [template](architecture/adr/_template.md)

## Dev
Working practices, priorities, and lessons learned.
- [Roadmap](dev/roadmap.md)
- [Working plan](dev/working_plan.md)
- [Code structure](dev/code_structure.md)
- [Lessons learned](dev/lessons_learned.md)
- [ML training dataset V2 workflow](dev/ml_training_dataset_v2.md)
- [Observation sub-block generator design (Phase 1 contract)](dev/obs_subblock_generator_design.md)
- Observation sub-block recipes:
  - `examples/recipes/observation_subblock_trace.py` (trace builder)
  - `examples/recipes/observation_subblock.py` (renderer)

## Tutorials
Hands-on guides for common workflows.
- [Canonical astrometry demo](tutorials/canonical_astrometry_demo.md)
- [Modeling overview](tutorials/modeling_overview.md)

## Archive
Retired references kept for historical context.
- [Legacy APIs and migration](archive/LEGACY_APIS_AND_MIGRATION.md)
- [Refactor history](archive/REFACTOR_HISTORY.md)
