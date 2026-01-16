# Testing architecture and runtime guide

This page is the source of truth for how the test suite is organized, what it exercises, and where the current runtime and duplication pain points are. Future test moves or refactors should update this document.

## Standard commands
- Full: `PYTHONPATH=src:. pytest -q`
- Fast (skip slow-marked integration paths): `PYTHONPATH=src:. pytest -q -m "not slow"`
- Timing sample: `PYTHONPATH=src:. pytest -q --durations=25`
  - Last run (January 16, 2026): 146 passed, 1 skipped in 309.00s (0:05:08).
  - Note: tests is now a package; required test command is `PYTHONPATH=src:. pytest …`.

## Marker policy
- `slow`: integration-heavy SHERA/Binder runs that dominate the runtime snapshot. Use `-m "not slow"` for a developer-speed pass; the full suite must still include them.
- Legacy SystemGraph coverage has been retired along with the graph layer; there are no legacy SystemGraph tests or markers in the active suite.
- Currently marked slow (collect-only on January 16, 2026):
  - `tests/binder/test_binder_smoke.py::test_shera_threeplane_binder_smoke`
  - `tests/inference/test_fim_theta.py::test_fim_theta_shape_and_symmetry`
  - `tests/inference/test_fim_theta.py::test_fim_theta_shera_wrapper_consistency`
  - `tests/inference/test_image_nll_bridge.py::test_run_image_gd_separation_smoke`
  - `tests/inference/test_inference_api.py::test_run_shera_image_gd_basic_separation_smoke`
  - `tests/inference/test_make_binder_nll_fn.py::test_theta0_store_override_keeps_binder_base_alignment`
  - `tests/inference/test_noiseless_truth_stationary.py::test_noiseless_truth_is_stationary_for_gaussian_nll`
  - `tests/model/test_model_builder.py::test_build_shera_threeplane_model_smoke`

## Shared fixtures
- `shera_smoke_cfg` / `shera_smoke_updates`: session fixtures for the standard SHERA testbed config and canonical parameter overrides (shared separation/position defaults plus zeroed Zernike vectors).
- `shera_smoke_forward` / `shera_smoke_inference`: session-scoped forward and inference `ParamSpec` + `ParameterStore` pairs built from the shared overrides, with deriveds refreshed once per session.
- `shera_smoke_binder_data` / `shera_smoke_model_data`: session-scoped synthetic PSF data and variance computed once via the Binder path and the model builder, respectively.
- `shera_smoke_infer_keys`: reusable tuple of the standard inference keys (`binary.separation_as`, `binary.x_position_as`, `binary.y_position_as`).

These fixtures are used by:
- `tests/inference/test_inference_api.py::test_run_shera_image_gd_basic_separation_smoke`
- `tests/inference/test_image_nll_bridge.py::*`
- `tests/inference/test_loss_canonical.py::test_loss_canonical_matches_binder_nll_and_is_jittable`
- `tests/inference/test_fim_theta.py::*`

### Timing snapshot
- Timestamp: January 16, 2026
- Command: `PYTHONPATH=src:. pytest -q --durations=25`
- Result: 146 passed, 1 skipped in 309.00s (0:05:08)
- Top 10 slowest tests:
  1. `tests/demos/test_demo_canonical_astrometry.py::test_canonical_astrometry_recipe_runs` — 140.26s call
  2. `tests/demos/test_twoplane_astrometry_demo.py::test_twoplane_astrometry_recipe_runs` — 53.42s call
  3. `tests/inference/test_inference_api.py::test_run_shera_image_gd_basic_separation_smoke` — 14.01s call
  4. `tests/inference/test_fim_theta.py::test_fim_theta_shape_and_symmetry` — 12.61s call
  5. `tests/inference/test_fim_theta.py::test_fim_theta_shera_wrapper_consistency` — 9.51s call
  6. `tests/inference/test_run_artifacts_integration.py::test_run_image_gd_writes_index_map_metadata` — 8.47s call
  7. `tests/binder/test_binder_smoke.py::test_shera_threeplane_binder_smoke` — 7.52s call
  8. `tests/inference/test_image_nll_bridge.py::test_run_image_gd_separation_smoke` — 7.50s call
  9. `tests/inference/test_image_nll_bridge.py::test_make_image_nll_fn_smoke_gaussian` — 6.13s setup
  10. `tests/binder/test_binder_diagnostics.py::test_binder_introspection_snapshot` — 4.91s call

## Current inventory (grouped by subject)

### Binder construction and accessors
- `tests/binder/test_binder_smoke.py`: three- and two-plane Binder smoke tests.
- `tests/binder/test_binder_shared_behaviour.py`: overlay/merge semantics, cfg/store passthrough.
- `tests/binder/test_binder_namespace.py`: namespace accessors and validation.
- `tests/binder/test_binder_leaf_access.py`, `tests/binder/test_binder_leaf_index.py`: leaf-path access, indexing helpers, and param path retrieval.
- `tests/binder/test_binder_dir.py`: testing new `__dir__` for tab completion.
- `tests/binder/test_binder_diagnostics.py`: diagnostics output structure.
- `tests/binder/test_binder_update_semantics.py`: update semantics and runtime overlay behavior.

### Parameters, packing, and store mechanics
- `tests/params/test_params_spec.py`, `tests/params/test_params_packing.py`: `ParamSpec` operations, pack/unpack round-trips and validation.
- `tests/params/test_params_store.py`, `tests/params/test_params_transforms.py`, `tests/params/test_store_namespace.py`: `ParameterStore` CRUD, transforms, and namespaced views.
- `tests/params/test_params_packing.py`, `tests/params/test_params_store.py`: error handling for missing keys, size mismatches, and inference subset shape validation.
- `tests/params/test_prior_spec.py`, `tests/params/test_refresh_derived_workflow.py`: prior spec definitions, derived value refresh workflow.
- Shared helpers live in `tests/conftest.py` for forward/inference store construction.

### Optics and modeling
- `tests/optics/test_optics_config.py`, `tests/optics/test_optics_builder.py`: optics configuration defaults, builder caching/miss/hit behavior.
- `tests/model/test_model_builder.py`: model construction smoke tests.
- `tests/optics/test_shera_threeplane_transforms.py`, `tests/optics/test_shera_twoplane_spec.py`: system-specific transform/spec wiring.
- `tests/optics/test_plate_scale_runtime.py`: runtime updates for plate scale and cached optics.
- `tests/model/test_universe_builder.py`: Alpha Cen source construction round-trip.

### Inference, losses, and optimization
- `tests/inference/test_image_nll_bridge.py`, `tests/inference/test_loss_canonical.py`, `tests/inference/test_fim_theta.py`, `tests/inference/test_inference_api.py`: end-to-end image NLL/FIM/gradient-descent smokes on SHERA configs.
- `tests/inference/test_run_eigen_gd.py`, `tests/inference/test_run_simple_gd.py`, `tests/inference/test_eigen_theta_map.py`: Eigenmode helpers, simple GD loops, and eigen map correctness.
- `tests/inference/test_losses.py`, `tests/inference/test_inference_helpers.py`, `tests/inference/test_make_binder_nll_fn.py`, `tests/inference/test_noiseless_truth_stationary.py`: loss helpers, inference spec validation, binder NLL construction, and stationary noise checks.
- `tests/inference/test_checkpoint_grad_diag.py`, `tests/inference/test_precond_artifacts.py`: checkpoint/diagnostic and preconditioner artifact coverage.
- `tests/inference/test_run_artifacts_integration.py`, `tests/inference/test_run_artifacts_io.py`: run output artifacts, index maps, and metadata IO.
- `tests/inference/test_plotting_smoke.py`, `tests/inference/test_signals.py`, `tests/inference/test_sweeps.py`: plotting smokes, signal utilities, and sweep helpers.

### Demos, plotting, and misc
- `tests/demos/test_demo_canonical_astrometry.py`, `tests/demos/test_twoplane_astrometry_demo.py`: canonical recipe/runner smoke tests in `fast` mode that assert outputs/plots.
- `tests/plotting/test_plotting.py`: plotting utilities including grid layout, PSF comparisons, and parameter history plots.
- `tests/devtools/test_generate_context_snapshot.py`: devtools context snapshot generation.
- `tests/devtools/test_imports.py`: package import smoke test.

## `tests/` taxonomy
- `tests/binder/`: Binder behavior, namespaces, diagnostics (`test_binder_*` files).
- `tests/params/`: specs, packing/unpacking, stores, transforms, priors, derived refresh.
- `tests/optics/`: optics config, builders, transforms.
- `tests/model/`: model builder, components, universe/source builders.
- `tests/inference/`: image NLL/FIM, GD helpers (simple and eigen), binder NLL, loss canonical, inference helpers.
- `tests/demos/`: demo script fast-mode checks.
- `tests/plotting/`: plotting utility coverage.
- `tests/devtools/`: context snapshot and other tooling smokes.
- Shared fixtures/helpers: `tests/conftest.py` centralizes the forward/inference store builders.

## Runtime hotspots (top 10)

| Duration (s) | Test |
| --- | --- |
| 140.26 (call) | `tests/demos/test_demo_canonical_astrometry.py::test_canonical_astrometry_recipe_runs` |
| 53.42 (call) | `tests/demos/test_twoplane_astrometry_demo.py::test_twoplane_astrometry_recipe_runs` |
| 14.01 (call) | `tests/inference/test_inference_api.py::test_run_shera_image_gd_basic_separation_smoke` |
| 12.61 (call) | `tests/inference/test_fim_theta.py::test_fim_theta_shape_and_symmetry` |
| 9.51 (call) | `tests/inference/test_fim_theta.py::test_fim_theta_shera_wrapper_consistency` |
| 8.47 (call) | `tests/inference/test_run_artifacts_integration.py::test_run_image_gd_writes_index_map_metadata` |
| 7.52 (call) | `tests/binder/test_binder_smoke.py::test_shera_threeplane_binder_smoke` |
| 7.50 (call) | `tests/inference/test_image_nll_bridge.py::test_run_image_gd_separation_smoke` |
| 6.13 (setup) | `tests/inference/test_image_nll_bridge.py::test_make_image_nll_fn_smoke_gaussian` |
| 4.91 (call) | `tests/binder/test_binder_diagnostics.py::test_binder_introspection_snapshot` |

## Repeated expensive setups
- SHERA synthetic data generation (`SHERA_TESTBED_CONFIG` + `ParameterStore` + Binder/model `.model()`): repeated in `tests/inference/test_inference_api.py`, `tests/inference/test_image_nll_bridge.py`, `tests/inference/test_loss_canonical.py`, `tests/inference/test_fim_theta.py`. Each re-JITs the same Binder/model build and produces fresh PSFs.
- 20-step gradient-descent loops (e.g., `run_image_gd`, `run_shera_image_gd_basic`, `run_shera_image_gd_eigen`): appear in `tests/inference/test_inference_api.py`, `tests/inference/test_image_nll_bridge.py`, `tests/inference/test_run_eigen_gd.py`, and `tests/inference/test_run_simple_gd.py`, all starting from similar stores and data.
- FIM and NLL construction with identical infer key sets (`binary.separation_as`, `binary.x_position_as`, `binary.y_position_as`) across `tests/inference/test_fim_theta.py`, `tests/inference/test_image_nll_bridge.py`, and `tests/inference/test_loss_canonical.py` regenerate the same `make_binder_image_nll_fn`/`make_image_nll_fn` closures.
- Demo smoke tests re-run SHERA builders in `fast` mode but still trigger Binder/model creation twice (`tests/demos/test_demo_canonical_astrometry.py`, `tests/demos/test_twoplane_astrometry_demo.py`).

## Future work: consolidation opportunities
- Shared fixtures exist in `tests/conftest.py`; most smokes should reuse them to avoid re-building configs/stores.
- Unify SHERA GD smoke harnesses (parameterize step counts, keep assertions minimal, mark high-res variants `slow`).
- Share NLL/FIM closure construction helpers across inference tests to encourage JAX cache hits.
- Reduce duplicate synthetic data generation by reusing session-scoped data fixtures where possible.
- Keep demo smoke tests strictly “fast mode” and avoid filesystem writes unless explicitly under a temp dir/fixture.
