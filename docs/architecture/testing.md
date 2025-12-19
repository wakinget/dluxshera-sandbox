# Testing architecture and runtime guide

This page is the source of truth for how the test suite is organized, what it exercises, and where the current runtime and duplication pain points are. Future test moves or refactors should update this document.

## Baseline command
- `PYTHONPATH=src:. pytest -q --durations=25`
  - Last run: 113 passed, 1 skipped in 763.93s (0:12:43).
  - Note: tests is now a package; required test command is `PYTHONPATH=src:. pytest …`.

## Current inventory (grouped by subject)

### Binder construction and accessors
- `tests/binder/test_binder_smoke.py`: three- and two-plane Binder smoke tests.
- `tests/binder/test_binder_shared_behaviour.py`: overlay/merge semantics, cfg/store passthrough.
- `tests/binder/test_binder_namespace.py`: namespace accessors and validation.
- `tests/binder/test_binder_leaf_access.py`, `tests/binder/test_binder_leaf_index.py`: leaf-path access, indexing helpers, and param path retrieval.
- `tests/binder/test_binder_dir.py`: testing new `__dir__` for tab completion.
- `tests/binder/test_binder_diagnostics.py`: diagnostics output structure.

### Parameters, packing, and store mechanics
- `tests/params/test_params_spec.py`, `tests/params/test_params_packing.py`: `ParamSpec` operations, pack/unpack round-trips and validation.
- `tests/params/test_params_store.py`, `tests/params/test_params_transforms.py`, `tests/params/test_store_namespace.py`: `ParameterStore` CRUD, transforms, and namespaced views.
- `tests/params/test_params_packing.py`, `tests/params/test_params_store.py`: error handling for missing keys, size mismatches, and inference subset shape validation.
- `tests/params/test_prior_spec.py`, `tests/params/test_refresh_derived_workflow.py`: prior spec definitions, derived value refresh workflow.
- Shared helpers live in `tests/conftest.py` (moved from `tests/helpers.py`) for forward/inference store construction.

### Optics, modeling, and graph
- `tests/optics/test_optics_config.py`, `tests/optics/test_optics_builder.py`: optics configuration defaults, builder caching/miss/hit behavior.
- `tests/model/test_model_builder.py`, `tests/model/test_modeling_components.py`: model construction smoke tests and component bundle validation.
- `tests/optics/test_system_graph.py`: system graph outputs vs. Binder parity and output mapping behavior.
- `tests/optics/test_shera_threeplane_transforms.py`, `tests/optics/test_shera_twoplane_spec.py`: system-specific transform/spec wiring.
- `tests/model/test_universe_builder.py`: Alpha Cen source construction round-trip.

### Inference, losses, and optimization
- `tests/inference/test_image_nll_bridge.py`, `tests/inference/test_loss_canonical.py`, `tests/inference/test_fim_theta.py`, `tests/inference/test_inference_api.py`: end-to-end image NLL/FIM/gradient-descent smokes on SHERA configs.
- `tests/inference/test_run_eigen_gd.py`, `tests/inference/test_run_simple_gd.py`, `tests/inference/test_eigen_theta_map.py`: Eigenmode helpers, simple GD loops, and eigen map correctness.
- `tests/inference/test_losses.py`, `tests/inference/test_inference_helpers.py`, `tests/inference/test_make_binder_nll_fn.py`, `tests/inference/test_noiseless_truth_stationary.py`: loss helpers, inference spec validation, binder NLL construction, and stationary noise checks.

### Demos, plotting, and misc
- `tests/demos/test_demo_canonical_astrometry.py`, `tests/demos/test_twoplane_astrometry_demo.py`: demo scripts run in `fast` mode and assert outputs/plots.
- `tests/plotting/test_plotting.py`: plotting utilities including grid layout, PSF comparisons, and parameter history plots.
- `tests/devtools/test_generate_context_snapshot.py`: devtools context snapshot generation.
- `tests/devtools/test_imports.py`: package import smoke test.

## `tests/` taxonomy
- `tests/binder/`: Binder behavior, namespaces, diagnostics (`test_binder_*` files).
- `tests/params/`: specs, packing/unpacking, stores, transforms, priors, derived refresh.
- `tests/optics/`: optics config, builders, transforms, system graphs.
- `tests/model/`: model builder, components, universe/source builders.
- `tests/inference/`: image NLL/FIM, GD helpers (simple and eigen), binder NLL, loss canonical, inference helpers.
- `tests/demos/`: demo script fast-mode checks.
- `tests/plotting/`: plotting utility coverage.
- `tests/devtools/`: context snapshot and other tooling smokes.
- Shared fixtures/helpers: `tests/conftest.py` centralizes the forward/inference store builders previously in `tests/helpers.py`.

### Path moves applied in Task 2
- `tests/test_binder_*.py` → `tests/binder/`
- `tests/test_params_*.py`, `tests/test_prior_spec.py`, `tests/test_refresh_derived_workflow.py`, `tests/test_store_namespace.py` → `tests/params/`
- `tests/test_optics_*.py`, `tests/test_shera_threeplane_transforms.py`, `tests/test_shera_twoplane_spec.py`, `tests/graph/test_system_graph.py` → `tests/optics/`
- `tests/test_model_builder.py`, `tests/test_modeling_components.py`, `tests/test_universe_builder.py` → `tests/model/`
- `tests/test_image_nll_bridge.py`, `tests/test_loss_canonical.py`, `tests/test_losses.py`, `tests/test_fim_theta.py`, `tests/test_inference_api.py`, `tests/test_run_eigen_gd.py`, `tests/test_run_simple_gd.py`, `tests/test_eigen_theta_map.py`, `tests/test_inference/*.py` → `tests/inference/`
- `tests/test_demo_canonical_astrometry.py`, `tests/test_twoplane_astrometry_demo.py` → `tests/demos/`
- `tests/test_plotting.py` → `tests/plotting/`
- `tests/test_generate_context_snapshot.py`, `tests/test_imports.py` → `tests/devtools/`
- `tests/helpers.py` → `tests/conftest.py` (import sites updated); path calculations in `tests/binder/test_binder_diagnostics.py` and `tests/demos/test_twoplane_astrometry_demo.py` now anchor to the repository root from their new locations.

## Runtime hotspots (top 25)

| Duration (s) | Test |
| --- | --- |
| 456.79 | `tests/test_inference_api.py::test_run_shera_image_gd_basic_separation_smoke` |
| 75.74 | `tests/test_fim_theta.py::test_fim_theta_shape_and_symmetry` |
| 52.34 | `tests/test_image_nll_bridge.py::test_run_image_gd_separation_smoke` |
| 46.75 | `tests/test_fim_theta.py::test_fim_theta_shera_wrapper_consistency` |
| 24.44 | `tests/test_loss_canonical.py::test_loss_canonical_matches_binder_nll_and_is_jittable` |
| 20.90 | `tests/test_inference/test_noiseless_truth_stationary.py::test_noiseless_truth_is_stationary_for_gaussian_nll` |
| 19.22 | `tests/test_inference/test_make_binder_nll_fn.py::test_theta0_store_override_keeps_binder_base_alignment` |
| 14.16 | `tests/test_image_nll_bridge.py::test_make_image_nll_fn_smoke_gaussian` |
| 9.92 | `tests/graph/test_system_graph.py::test_system_graph_forward_matches_legacy_model` |
| 6.01 | `tests/test_model_builder.py::test_build_shera_threeplane_model_smoke` |
| 5.92 | `tests/test_demo_canonical_astrometry.py::test_canonical_astrometry_demo_runs` |
| 4.86 | `tests/test_binder_smoke.py::test_shera_threeplane_binder_smoke` |
| 4.17 | `tests/test_image_nll_bridge.py::test_make_binder_image_nll_fn_smoke_gaussian` |
| 2.41 | `tests/test_run_eigen_gd.py::test_eigen_and_pure_theta_share_binder_loss` |
| 2.13 | `tests/test_run_eigen_gd.py::test_eigen_helper_quadratic_roundtrip_and_descent` |
| 2.01 | `tests/test_loss_canonical.py::test_make_binder_image_nll_fn_twoplane_smoke` |
| 1.49 | `tests/test_plotting.py::test_plot_parameter_history_grid` |
| 1.20 | `tests/test_twoplane_astrometry_demo.py::test_twoplane_astrometry_demo_runs` |
| 1.20 | `tests/test_run_eigen_gd.py::test_run_shera_image_gd_eigen_smoke` |
| 1.12 | `tests/test_run_simple_gd.py::test_run_simple_gd_converges_on_quadratic` |
| 1.06 | `tests/test_optics_builder.py::test_threeplane_optics_cache_miss_on_structural_change` |
| 0.85 | `tests/test_eigen_theta_map.py::test_eigen_theta_map_whitened_scales_quadratic` |
| 0.82 | `tests/test_plotting.py::test_plot_psf_comparison_grid` |
| 0.57 | `tests/test_eigen_theta_map.py::test_eigen_theta_map_roundtrip_unwhitened` |
| 0.57 | `tests/test_optics_builder.py::test_threeplane_optics_cache_hits` |

## Repeated expensive setups
- SHERA synthetic data generation (`SHERA_TESTBED_CONFIG` + `ParameterStore` + Binder/model `.model()`): repeated in `test_inference_api.py`, `test_image_nll_bridge.py`, `test_loss_canonical.py`, `test_fim_theta.py`, and parts of `tests/graph/test_system_graph.py`. Each re-JITs the same Binder/model build and produces fresh PSFs.
- 20-step gradient-descent loops (e.g., `run_image_gd`, `run_shera_image_gd_basic`, `run_shera_image_gd_eigen`): appear in `test_inference_api.py`, `test_image_nll_bridge.py`, `test_run_eigen_gd.py`, and `test_run_simple_gd.py`, all starting from similar stores and data.
- FIM and NLL construction with identical infer key sets (`binary.separation_as`, `binary.x_position_as`, `binary.y_position_as`) across `test_fim_theta.py`, `test_image_nll_bridge.py`, and `test_loss_canonical.py` regenerate the same `make_binder_image_nll_fn`/`make_image_nll_fn` closures.
- Demo smoke tests re-run SHERA builders in `fast` mode but still trigger Binder/model creation twice (`test_demo_canonical_astrometry.py`, `test_twoplane_astrometry_demo.py`).

## Prioritized consolidation plan
1) **Centralize SHERA testbed fixtures:** Create `tests/conftest.py` session-scoped fixtures that return `(cfg, forward_spec, forward_store, inference_spec, inference_store, binder/model outputs, synthetic data/var)`. Reuse in inference/FIM/graph tests to avoid repeat builds and JITs of identical configs.
2) **Share gradient-descent harnesses:** Provide a small fixture or helper that returns a precomputed `(loss_fn, theta0, truth_store)` tuple for SHERA runs. Let GD-centric tests consume it with parameterized step counts to keep assertions while shortening loops; mark the longest variants as `@pytest.mark.slow`.
3) **Unify NLL/FIM construction:** Extract a helper that constructs `make_binder_image_nll_fn`/`make_image_nll_fn` with standard infer keys and synthetic data. Use it across `test_fim_theta`, `test_image_nll_bridge`, `test_loss_canonical`, and `test_inference_api` to collapse duplicate setup blocks and enable caching of compiled closures.
4) **Isolate demo exercises:** Move demo smoke tests under `tests/demos/` and gate plots via fixtures to avoid repeated filesystem writes; consider a shared “fast demo” fixture that produces the cached PSFs once per session.
