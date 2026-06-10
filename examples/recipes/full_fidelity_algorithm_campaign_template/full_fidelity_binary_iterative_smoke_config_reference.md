# Full-Fidelity Binary Iterative Smoke Config Reference

Generated field reference for the executable smoke schema. The future `full_fidelity_algorithm_campaign_v1.yaml` skeleton is intentionally not covered exhaustively here; do not copy its future-only blocks into the smoke config without adding runner support.

| Field path | Example value | Required? | Consumed by | Runtime effect | Fidelity effect | Provenance effect | Safe to omit? | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| experiment.kind | full_fidelity_binary_iterative_smoke | True | wrapper | selects translator | none | records source schema | False | Must be full_fidelity_binary_iterative_smoke for executable smoke configs. |
| experiment.schema_version | full_fidelity_binary_iterative_smoke.v1 | True | wrapper/provenance | none | none | schema label | False | Used by reviewers to distinguish the smoke schema from the future skeleton. |
| experiment.run_name | full_fidelity_binary_iterative_smoke | False | wrapper | output path only | none | run identity | True | CLI --run-name overrides this value. |
| experiment.seed | 42 | False | wrapper/observation-bias | deterministic seeding | changes random realization | base seed | True | Default is 42 in the wrapper. |
| experiment.source_kind | binary_target | False | wrapper | source resolver choice | selects binary target source semantics | source label | True | Defaults to binary_target. |
| experiment.target | ALPHA_CEN | False | wrapper/model-split | SED/source lookup | target-aware spectral deck | science target | True | Defaults to ALPHA_CEN. |
| experiment.n_cases | 1 | False | wrapper | case count when prior_draws omitted | none | smoke scale | True | Keep 1 for smoke; use 2-4 for a less tiny validation. |
| experiment.n_draws | 1 | False | smoke label only | none today | none | documents intended tiny draw count | True | prior_draws.n_cases controls generated prior cases. |
| experiment.system_preset | SHERA_FLIGHT_3P | False | wrapper | system resolver | selects SHERA preset | system label | True | Defaults to SHERA_FLIGHT_3P. |
| experiment.spectral_model.enabled | True | False | model-split helper | enables spectral deck | truth/reference spectral mismatch possible | component summary | True | If false/omitted, spectral model split is disabled. |
| experiment.spectral_model.fast | True | False | model-split helper | clamps truth<=7 and inference<=5 wavelengths | reduces spectral sampling fidelity | smoke shortcut | True | Not a substitute for explicit n_lambda/range/response settings. |
| experiment.spectral_model.preserve_flux_parameters | True | False | model-split helper | none significant | keeps scalar flux parameters separate from spectral weights | spectral config | True | Default is true in model-split composition. |
| experiment.spectral_model.source_seds.mode | target | False | model-split helper | SED lookup | target-aware SED selection | spectral source mode | True | target is the current smoke path. |
| experiment.spectral_model.truth.n_lambda | 51 | False | model-split helper | truth wavelength grid cost | truth spectral sampling | truth deck | True | With fast=true the effective value is clamped to <=7. |
| experiment.spectral_model.truth.components.detector_qe.enabled | False | False | model-split helper | response-table work if enabled | detector QE realism | truth component | True | Disabled in smoke to keep dependencies/cost small. |
| experiment.spectral_model.truth.components.m2_filter_response.enabled | False | False | model-split helper | response-table work if enabled | filter response realism | truth component | True | Disabled in smoke to keep dependencies/cost small. |
| experiment.spectral_model.inference.n_lambda | 3 | False | model-split helper | inference/reference spectral grid cost | reference spectral sampling | inference deck | True | Use 5-7 for a less tiny validation; fast=true clamps to <=5. |
| experiment.high_order_wfe.truth.npix | 16 | False | model-split helper | WFE map generation and optics array size | high-order spatial sampling | truth WFE deck | True | 16 is smoke scale; 32 or 64 is a next-step validation value. |
| experiment.high_order_wfe.truth.amplitude_nm_rms | 0.3 | False | model-split helper | none significant | truth high-order WFE amplitude | truth WFE deck | True | Controls physical truth/reference mismatch when knowledge_error is nonzero. |
| experiment.high_order_wfe.artifacts.write_maps | False | False | model-split helper | artifact I/O | none | debug artifact availability | True | False keeps smoke artifacts small. |
| experiment.subblocks.n_frames | 3 | False | observation-bias campaign | linear-ish render/inference cost | temporal sampling | plan scale | True | 3 is smoke scale; 5-10 is a less tiny validation. |
| experiment.subblocks.reference_n_iter | 3 | False | observation-bias campaign | optimizer iterations | reference-solve convergence | optimizer config | True | 3 is deliberately tiny. |
| experiment.subblocks.trajectory_processing.smear.render.mode | metadata_only | False | model-split metadata | metadata_only avoids dynamic smear rendering | dynamic smear not active | smear sidecar mode | True | Only none/metadata_only are wired in the smoke wrapper. |
| experiment.iterative.update_gain | 0.25 | False | observation-bias iterative update | none significant | update damping/stability | iterative plan | True | 0.25 is conservative for smoke. |
| experiment.iterative.update_safety.posterior_sigma_inflation | 10.0 | False | observation-bias iterative update | none significant | conservative posterior uncertainty | update safety | True | 10.0 is a smoke safety guard. |
| experiment.observation_theta.optics.primary_zernikes.indices | [0] | False | observation theta layout | state dimension | active low-order optical parameters | theta layout | True | [0] is tiny; from_system or more indices increases state dimension. |
| experiment.prior_draws.sigmas.* |  | False | observation-bias campaign | prior draw generation | initial offset size | prior draw table | True | Initialization only; does not add physical realism terms. |
