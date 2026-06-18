"""Registry-backed contract metadata for full-fidelity executable configs."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Mapping

IMPLEMENTED_STATUSES = (
    "implemented",
    "accepted_but_noop",
    "smoke_only",
    "future_placeholder",
    "deprecated",
    "unknown_should_fail",
)


def _entry(
    *,
    valid_values: Mapping[str, Mapping[str, str]] | None = None,
    default: Any = None,
    implemented_status: str = "implemented",
    consumed_by: str = "documented contract",
    runtime_effect: str = "none",
    fidelity_effect: str = "none",
    provenance_effect: str = "none",
    safe_to_omit: bool = True,
    notes: str = "",
) -> dict[str, Any]:
    if implemented_status not in IMPLEMENTED_STATUSES:
        raise ValueError(f"Bad implemented_status={implemented_status!r}")
    return {
        "valid_values": dict(valid_values or {}),
        "default": default,
        "implemented_status": implemented_status,
        "consumed_by": consumed_by,
        "runtime_effect": runtime_effect,
        "fidelity_effect": fidelity_effect,
        "provenance_effect": provenance_effect,
        "safe_to_omit": safe_to_omit,
        "notes": notes,
    }


def _values(*items: tuple[str, str, str]) -> dict[str, dict[str, str]]:
    return {value: {"status": status, "description": desc} for value, status, desc in items}


CONFIG_FIELD_REGISTRY: dict[str, dict[str, Any]] = {
    "experiment.kind": _entry(
        valid_values=_values(
            ("full_fidelity_binary_iterative", "implemented", "Canonical executable full-fidelity binary iterative schema translated to observation_bias_campaign."),
            ("full_fidelity_binary_iterative_review", "deprecated_alias", "Deprecated compatibility alias normalized to full_fidelity_binary_iterative."),
            ("full_fidelity_binary_iterative_smoke", "deprecated_alias", "Deprecated compatibility alias normalized to full_fidelity_binary_iterative."),
            ("observation_bias_campaign", "implemented", "Already translated campaign config."),
            ("full_fidelity_algorithm_campaign", "future_placeholder", "Future schema skeleton; intentionally non-executable."),
        ),
        consumed_by="full-fidelity wrapper",
        runtime_effect="selects translator/validation contract",
        safe_to_omit=False,
    ),
    "experiment.schema_version": _entry(consumed_by="wrapper/provenance", provenance_effect="schema identity", safe_to_omit=False),
    "experiment.run_name": _entry(consumed_by="wrapper", runtime_effect="output path identity", provenance_effect="run identity"),
    "experiment.source_kind": _entry(valid_values=_values(("binary_target", "implemented", "Target-aware binary source."), ("binary", "implemented", "Generic binary source where supported."), ("alpha_cen", "implemented", "Alpha-Cen compatibility alias.")), consumed_by="wrapper/model split", fidelity_effect="source builder selection"),
    "experiment.target": _entry(consumed_by="source/spectral resolver", fidelity_effect="target SED/source selection"),
    "experiment.system_preset": _entry(valid_values=_values(("SHERA_FLIGHT_3P_CONV", "implemented", "Full-fidelity default preset with named detector realism layers."), ("SHERA_FLIGHT_3P", "implemented", "Legacy flight three-plane baseline preset."), ("SHERA_FLIGHT_2P", "implemented", "Two-plane flight preset."), ("SHERA_TESTBED_3P", "implemented", "Testbed three-plane preset.")), default="SHERA_FLIGHT_3P_CONV", consumed_by="config resolver", fidelity_effect="base source/optics/detector stack"),
    "experiment.detector_overrides.layers.*.action": _entry(valid_values=_values(("keep", "implemented", "Leave named layer unchanged."), ("update", "implemented", "Deep-merge fields into a named layer."), ("remove", "implemented", "Remove a named layer."), ("disable", "implemented", "Alias for remove.")), consumed_by="detector layer override utility", fidelity_effect="patches resolved detector layer stack"),
    "experiment.detector_overrides.layers.*.allow_missing": _entry(consumed_by="detector layer override utility", runtime_effect="missing named layer does not fail when true"),
    "experiment.detector_overrides.layers.*.kernel.kind": _entry(valid_values=_values(("gaussian", "implemented", "Gaussian convolution kernel."), ("box", "implemented", "Box/pixel-MTF convolution kernel."), ("line", "implemented", "Line-smear convolution kernel.")), consumed_by="detector layer override utility"),
    "experiment.detector_overrides.layers.*.kernel.units": _entry(valid_values=_values(("detector_pix", "implemented", "Detector pixel units.")), consumed_by="detector layer override utility"),
    "experiment.n_cases": _entry(consumed_by="wrapper fallback", runtime_effect="case count if prior_draws omitted"),
    "experiment.n_draws": _entry(implemented_status="smoke_only", consumed_by="review label only", notes="prior_draws.n_cases controls generated cases."),
    "experiment.spectral_model.enabled": _entry(consumed_by="campaign_model_split", runtime_effect="enables spectral deck", fidelity_effect="truth/reference chromatic model"),
    "experiment.spectral_model.fast": _entry(implemented_status="smoke_only", consumed_by="campaign_model_split", runtime_effect="clamps truth<=7 and inference<=5 wavelengths", fidelity_effect="reduces spectral fidelity", notes="Allowed only in smoke config; absent from review config."),
    "experiment.spectral_model.preserve_flux_parameters": _entry(consumed_by="spectral source patcher", fidelity_effect="keeps log_flux_total/contrast as detected band-integrated scalar parameters", provenance_effect="records preserved scalar policy"),
    "experiment.spectral_model.photometry_mode": _entry(
        valid_values=_values(
            ("preserve_detected_flux_parameters", "implemented", "Equivalent to preserve_flux_parameters=true; throughput affects normalized weights and provenance flux factors only."),
            ("apply_total_throughput_to_log_flux", "future_placeholder", "Would modify total scalar flux by throughput."),
            ("apply_component_throughput_to_log_flux_and_contrast", "future_placeholder", "Would modify total flux and contrast by component throughput."),
            ("reference_band_calibrated", "future_placeholder", "Would calibrate scalar photometry to an explicit reference band."),
        ),
        default="preserve_detected_flux_parameters",
        consumed_by="wrapper validation/model split policy",
        fidelity_effect="scalar flux semantics",
        notes="Only preserve_detected_flux_parameters is supported today.",
    ),
    "experiment.spectral_model.source_seds.mode": _entry(valid_values=_values(("target", "implemented", "Resolve component SEDs from source.target."), ("real", "implemented", "Compatibility alias for target."), ("flat", "implemented", "Flat unit SED fallback in spectral utilities."), ("explicit", "future_placeholder", "Use explicit_paths only; partial support exists via utilities but not review contract default.")), consumed_by="spectral resolver", fidelity_effect="source SED selection"),
    "experiment.spectral_model.source_seds.generic_binary_fallback": _entry(valid_values=_values(("alpha_cen", "implemented", "Allow generic binary to use Alpha Cen component SEDs."), ("require_explicit", "implemented", "Require explicit component SED paths.")), consumed_by="spectral resolver"),
    "experiment.spectral_model.truth.label": _entry(consumed_by="spectral provenance", provenance_effect="truth spectral deck label"),
    "experiment.spectral_model.truth.mode": _entry(valid_values=_values(("effective_source_spectrum", "implemented", "SED times enabled response components sampled on truth wavelength grid.")), default="effective_source_spectrum", consumed_by="audit/schema", notes="Currently accepted as documentation; model_split builds this mode implicitly."),
    "experiment.spectral_model.truth.components.detector_qe.enabled": _entry(consumed_by="spectral resolver", runtime_effect="loads/interpolates QE curve", fidelity_effect="detector QE throughput in weights"),
    "experiment.spectral_model.truth.components.m2_filter_response.enabled": _entry(consumed_by="spectral resolver", runtime_effect="loads/interpolates filter curve", fidelity_effect="M2/filter response in weights"),
    "experiment.spectral_model.truth.components.*.path": _entry(consumed_by="spectral resolver", runtime_effect="loads response CSV"),
    "experiment.spectral_model.truth.components.*.label": _entry(consumed_by="spectral provenance", provenance_effect="response component label"),
    "experiment.spectral_model.truth.components.*.wavelength_column": _entry(consumed_by="spectral resolver", runtime_effect="CSV wavelength column name"),
    "experiment.spectral_model.truth.components.*.wavelength_unit": _entry(valid_values=_values(("nm", "implemented", "Nanometres."), ("m", "implemented", "Metres."), ("um", "implemented", "Microns."), ("angstrom", "implemented", "Angstroms.")), consumed_by="spectral resolver"),
    "experiment.spectral_model.truth.components.*.response_column": _entry(consumed_by="spectral resolver", runtime_effect="CSV response column name"),
    "experiment.spectral_model.truth.components.*.response_unit": _entry(valid_values=_values(("dimensionless", "implemented", "Unitless response."), ("percent_reflection", "implemented", "Percent reflectance scaled by response_scale."), ("percent_transmission", "implemented", "Percent transmission scaled by response_scale.")), consumed_by="spectral provenance"),
    "experiment.spectral_model.truth.components.*.detector_model_proxy_for": _entry(consumed_by="spectral provenance", provenance_effect="records detector proxy assumption"),
    "experiment.spectral_model.truth.components.*.assumption": _entry(consumed_by="spectral provenance", provenance_effect="records response-data assumption"),
    "experiment.spectral_model.inference.label": _entry(consumed_by="spectral provenance", provenance_effect="inference spectral deck label"),
    "experiment.spectral_model.inference.mode": _entry(valid_values=_values(("effective_source_spectrum", "implemented", "Build reference effective spectrum on inference grid."), ("truncated_effective_source_spectrum", "implemented", "Review label for inference grid possibly narrower than truth.")), default="effective_source_spectrum", consumed_by="audit/schema", notes="model_split uses wavelength fields and out_of_band_response rather than branching on this label."),
    "experiment.spectral_model.inference.out_of_band_response": _entry(valid_values=_values(("zero", "implemented", "Out-of-band response contributes zero before renormalization."), ("edge_hold", "future_placeholder", "Future edge extrapolation policy."), ("extrapolate", "future_placeholder", "Future extrapolation policy."), ("error", "future_placeholder", "Future strict out-of-band failure policy.")), default="zero", consumed_by="spectral resolver", fidelity_effect="truncated reference spectral weights"),
    "experiment.spectral_model.inference.renormalize_weights": _entry(consumed_by="spectral resolver", fidelity_effect="normalizes inference weights after truncation"),
    "experiment.spectral_model.inference.components.detector_qe.mode": _entry(valid_values=_values(("same_as_truth", "implemented", "Use the truth response component definition."), ("disabled", "future_placeholder", "Disable response in inference only."), ("explicit_response_file", "future_placeholder", "Use a separate inference response file."), ("synthetic_flat", "future_placeholder", "Use flat response."), ("knowledge_error", "future_placeholder", "Perturb truth response by a calibration error.")), consumed_by="campaign_model_split"),
    "experiment.spectral_model.inference.components.m2_filter_response.mode": _entry(valid_values=_values(("same_as_truth", "implemented", "Use the truth response component definition."), ("disabled", "future_placeholder", "Disable response in inference only."), ("explicit_response_file", "future_placeholder", "Use a separate inference response file."), ("synthetic_flat", "future_placeholder", "Use flat response."), ("knowledge_error", "future_placeholder", "Perturb truth response by a calibration error.")), consumed_by="campaign_model_split"),
    "experiment.high_order_wfe.enabled": _entry(consumed_by="campaign_high_order_wfe", runtime_effect="generates OPD maps", fidelity_effect="high-order WFE realism"),
    "experiment.high_order_wfe.truth.mode": _entry(valid_values=_values(("synthetic", "implemented", "Generate synthetic power-law OPD maps.")), consumed_by="campaign_high_order_wfe"),
    "experiment.high_order_wfe.truth.mirrors": _entry(valid_values=_values(("primary", "implemented", "Primary mirror map."), ("secondary", "implemented", "Secondary mirror map.")), consumed_by="campaign_high_order_wfe"),
    "experiment.high_order_wfe.truth.pairing": _entry(valid_values=_values(("independent", "implemented", "Generate independent primary/secondary maps."), ("shared", "future_placeholder", "Future shared/common map policy.")), consumed_by="campaign_high_order_wfe"),
    "experiment.high_order_wfe.truth.remove_low_order_zernikes": _entry(consumed_by="campaign_high_order_wfe", fidelity_effect="removes configured low-order modes from high-order truth residual"),
    "experiment.high_order_wfe.truth.remove_zernike_modes": _entry(consumed_by="campaign_high_order_wfe", fidelity_effect="explicit low-order removal list"),
    "experiment.high_order_wfe.inference.mode": _entry(valid_values=_values(("knowledge_error", "implemented", "Reference map equals truth high-order map plus additive correlated error."), ("same_as_truth", "future_placeholder", "Matched reference map.")), consumed_by="campaign_high_order_wfe"),
    "experiment.high_order_wfe.inference.use_truth_common_map": _entry(implemented_status="accepted_but_noop", consumed_by="audit/schema", notes="Current implementation always uses the generated truth map as the reference base."),
    "experiment.high_order_wfe.inference.knowledge_error.enabled": _entry(consumed_by="campaign_high_order_wfe", fidelity_effect="truth/reference high-order mismatch"),
    "experiment.high_order_wfe.inference.knowledge_error.power_law_alpha": _entry(valid_values=_values(("same_as_truth", "implemented", "Use truth power-law alpha for error map.")), consumed_by="campaign_high_order_wfe", notes="Numeric values are also implemented."),
    "experiment.high_order_wfe.inference.knowledge_error.remove_low_order_zernikes": _entry(consumed_by="campaign_high_order_wfe", fidelity_effect="removes low-order Zernike projection from additive error"),
    "experiment.high_order_wfe.inference.knowledge_error.realization_policy": _entry(valid_values=_values(("additive_correlated", "implemented", "Generate additive correlated high-order error."), ("white", "future_placeholder", "Future white-error option.")), default="additive_correlated", consumed_by="campaign_high_order_wfe"),
    "experiment.high_order_wfe.validation.require_nonzero_difference_when_enabled": _entry(consumed_by="campaign_high_order_wfe", runtime_effect="validation guard"),
    "experiment.high_order_wfe.validation.max_abs_low_order_projection_nm": _entry(consumed_by="campaign_high_order_wfe", runtime_effect="warn/fail if additive error has low-order leakage"),
    "experiment.high_order_wfe.artifacts.write_maps": _entry(consumed_by="campaign_high_order_wfe", runtime_effect="writes FITS/NPY maps", provenance_effect="map artifacts"),
    "experiment.high_order_wfe.artifacts.write_png_quicklooks": _entry(implemented_status="accepted_but_noop", consumed_by="campaign_high_order_wfe", notes="PNG quicklooks are not currently written by the helper."),
    "experiment.high_order_wfe.artifacts.write_summary_json": _entry(consumed_by="campaign_high_order_wfe", provenance_effect="summary JSON artifact"),
    "experiment.subblocks.noise": _entry(valid_values=_values(("disabled", "implemented", "Legacy scalar: no render noise."), ("enabled", "implemented", "Legacy scalar: enable runner noise path."), ("inherit", "implemented", "Subblock runner default/inherited mode.")), consumed_by="observation_bias/subblock runner", notes="Structured mappings are translated to enabled/disabled plus provenance."),
    "experiment.subblocks.noise.enabled": _entry(consumed_by="wrapper translation", runtime_effect="render noise on/off", provenance_effect="normalized request", notes="Structured replacement for legacy scalar enabled/disabled."),
    "experiment.subblocks.noise.shot_noise": _entry(default=True, consumed_by="review audit/noise utility", runtime_effect="signal-dependent shot noise", provenance_effect="normalized request", notes="Also written as photon_noise for legacy render-template compatibility."),
    "experiment.subblocks.noise.read_noise": _entry(default=False, consumed_by="review audit/noise utility", runtime_effect="render read noise", provenance_effect="detector/spec provenance", notes="Current wrapper records this term and maps to the coarse runner flag; notebook audit applies it through the shared noise utility."),
    "experiment.subblocks.noise.dark_current": _entry(default=False, consumed_by="review audit/noise utility", runtime_effect="render dark-current variance", provenance_effect="detector/spec provenance", notes="Current wrapper records this term and maps to the coarse runner flag; notebook audit applies it through the shared noise utility."),
    "experiment.subblocks.noise.use_detector_read_noise": _entry(default=True, consumed_by="review audit/noise utility", runtime_effect="amplitude resolution", provenance_effect="detector/spec provenance", notes="Disable only when an explicit read_noise_electrons override is provided or read noise is off."),
    "experiment.subblocks.noise.read_noise_electrons": _entry(default=None, consumed_by="review audit/noise utility", runtime_effect="read-noise amplitude override", provenance_effect="config_override when set", notes="If null and read noise is enabled, detector/spec value is resolved and recorded."),
    "experiment.subblocks.noise.use_detector_dark_current": _entry(default=True, consumed_by="review audit/noise utility", runtime_effect="amplitude resolution", provenance_effect="detector/spec provenance", notes="Disable only when an explicit dark_current_e_per_s override is provided or dark current is off."),
    "experiment.subblocks.noise.dark_current_e_per_s": _entry(default=None, consumed_by="review audit/noise utility", runtime_effect="dark-current amplitude override", provenance_effect="config_override when set", notes="If null and dark current is enabled, detector/spec value is resolved and recorded."),
    "experiment.subblocks.noise.variance_floor": _entry(default=1.0, consumed_by="inference variance model", runtime_effect="likelihood variance floor", provenance_effect="inference provenance", notes="Inference-side floor; not silently applied to render variance."),
    "experiment.subblocks.noise.write_variance": _entry(default=True, consumed_by="render audit/provenance", runtime_effect="variance artifact request", provenance_effect="render provenance", notes="Audit warns if requested but no variance artifact/map is produced."),
    "experiment.subblocks.noise.seed_policy": _entry(valid_values=_values(("from_subblock_noise_seed", "implemented", "Use derived subblock noise seed.")), consumed_by="wrapper translation"),
    "experiment.subblocks.use_render_variance": _entry(valid_values=_values(("auto", "implemented", "Report template/default behavior and use rendered variance when the runner can prove one exists."), ("true", "implemented", "Require rendered variance cube for provided_cube inference model."), ("false", "implemented", "Use data/floor inference variance model.")), default="auto", consumed_by="inference template/audit", runtime_effect="likelihood variance source", provenance_effect="inference provenance", notes="Kept separate from render/data noise controls."),
    "experiment.subblocks.phi_ref": _entry(valid_values=_values(("truth_when_available", "implemented", "Use truth fast-state reference where available."), ("recovered", "implemented", "Use recovered-reference inference."), ("init", "future_placeholder", "Initialization reference where supported.")), consumed_by="subblock runner"),
    "experiment.subblocks.schur_curvature_method": _entry(valid_values=_values(("auto", "implemented", "Select available curvature path."), ("dense", "implemented", "Dense curvature path where dimensions allow."), ("structured", "implemented", "Structured curvature path.")), consumed_by="subblock runner"),
    "experiment.subblocks.reference_diagnostics_profile": _entry(valid_values=_values(("none", "implemented", "Disable extra diagnostics."), ("basic", "implemented", "Basic diagnostics."), ("review", "implemented", "Review diagnostics and plots."), ("full", "implemented", "Full expensive diagnostics.")), consumed_by="subblock runner"),
    "experiment.subblocks.reference_optimizer_kind": _entry(valid_values=_values(("sgd", "implemented", "SGD recovered-reference optimizer."), ("adam", "implemented", "Adam recovered-reference optimizer.")), consumed_by="subblock runner"),
    "experiment.subblocks.reference_schedule_kind": _entry(valid_values=_values(("constant", "implemented", "Constant LR."), ("linear_warmup", "implemented", "Linear warmup."), ("piecewise_constant", "implemented", "Piecewise schedule."), ("exponential_decay", "implemented", "Exponential decay."), ("cosine_decay", "implemented", "Cosine decay."), ("linear_warmup_cosine_decay", "implemented", "Warmup then cosine.")), consumed_by="subblock runner"),
    "experiment.subblocks.summary_information_scale": _entry(valid_values=_values(("summed_likelihood", "implemented", "Information matrix represents summed likelihood."), ("optimizer", "implemented", "Use optimizer-scale information.")), consumed_by="observation_bias/subblock runner"),
    "experiment.subblocks.schur_frame_quality_policy": _entry(valid_values=_values(("warn", "implemented", "Warn on bad frames."), ("mask", "implemented", "Mask bad frames."), ("reject", "implemented", "Reject subblock.")), consumed_by="subblock runner"),
    "experiment.subblocks.schur_frame_quality_missing": _entry(valid_values=_values(("allow_all", "implemented", "Allow when quality diagnostics missing."), ("error", "implemented", "Error when missing.")), consumed_by="subblock runner"),
    "experiment.subblocks.schur_frame_mask_denominator": _entry(valid_values=_values(("original", "implemented", "Use original frame count denominator."), ("kept", "implemented", "Use kept-frame denominator.")), consumed_by="subblock runner"),
    "experiment.subblocks.trace_source.mode": _entry(valid_values=_values(("trajectory", "implemented", "Materialize frame truth from trajectory."), ("iid_jitter", "implemented", "Legacy IID trace template."), ("external_plan", "implemented", "Use external frame truth plan.")), consumed_by="campaign_trace_sources"),
    "experiment.subblocks.trace_source.source.kind": _entry(valid_values=_values(("airbus_csv", "implemented", "Backward-compatible Airbus XYZ arcsec CSV."), ("csv", "implemented", "Generic CSV alias when format=airbus_xyz_arcsec.")), consumed_by="campaign_trace_sources"),
    "experiment.subblocks.trace_source.source.format": _entry(valid_values=_values(("airbus_xyz_arcsec", "implemented", "CSV columns/time,X,Y,Z in arcsec with Z converted to PA degrees.")), consumed_by="campaign_trace_sources"),
    "experiment.subblocks.trace_source.source.path": _entry(consumed_by="campaign_trace_sources", runtime_effect="trajectory CSV source path", fidelity_effect="trajectory segment source"),
    "experiment.subblocks.trace_source.sampling.interpolation": _entry(valid_values=_values(("linear", "implemented", "Linear interpolation.")), consumed_by="obs_subblock_trajectory"),
    "experiment.subblocks.trajectory_processing.high_pass.method": _entry(valid_values=_values(("none", "implemented", "No production high-pass filter."), ("exponential_lowpass_subtract", "future_placeholder", "Future 15 s high-pass production filter."), ("rolling_mean_subtract", "future_placeholder", "Future rolling mean high-pass.")), consumed_by="audit/schema", notes="Notebook diagnostic is separate from production trace source."),
    "experiment.subblocks.trajectory_processing.high_pass.note": _entry(consumed_by="documentation", provenance_effect="documents deferred high-pass intent"),
    "experiment.subblocks.trajectory_processing.smear.render.mode": _entry(valid_values=_values(("disabled", "implemented", "Remove/disable the named smear layer and write no rendered smear."), ("none", "implemented", "Compatibility alias for no rendered smear."), ("metadata_only", "implemented", "Diagnostic sidecars only; remove/disable the named smear render layer."), ("subblock_constant_layer", "implemented", "Patch a subblock fit-derived one-frame-exposure line kernel into the named smear layer."), ("per_frame", "future_placeholder", "Future dynamic per-frame kernel rendering.")), consumed_by="trajectory_smear"),
    "experiment.subblocks.trajectory_processing.smear.render.target_layer": _entry(consumed_by="trajectory_smear", default="smear", runtime_effect="selects the named detector layer to patch/remove"),
    "experiment.subblocks.trajectory_processing.smear.render.require_existing_layer": _entry(consumed_by="trajectory_smear", default=True, runtime_effect="fails when subblock_constant_layer target is absent"),
    "experiment.subblocks.trajectory_processing.smear.render.allow_layer_injection": _entry(consumed_by="trajectory_smear", default=False, runtime_effect="allows opt-in insertion of a missing smear layer"),
    "experiment.subblocks.trajectory_processing.smear.render.defaults.units": _entry(valid_values=_values(("detector_pix", "implemented", "Detector pixel units.")), consumed_by="trajectory_smear"),
    "experiment.subblocks.trajectory_processing.smear.inference.mode": _entry(valid_values=_values(("matched_subblock_constant", "implemented", "Patch the inference template with the same subblock smear kernel as truth/render."), ("disabled", "implemented", "Remove/disable smear in the inference/reference template."), ("solve_subblock_smear", "future_placeholder", "Future nuisance-parameter smear solve."), ("per_frame", "future_placeholder", "Future per-frame inference smear."), ("mismatch_scaled", "future_placeholder", "Future length mismatch policy."), ("mismatch_angle_offset", "future_placeholder", "Future direction mismatch policy.")), consumed_by="trajectory_smear"),
    "experiment.subblocks.trace_source.processing.filter.kind": _entry(valid_values=_values(("high_pass", "implemented", "High-pass trajectory filtering.")), consumed_by="campaign_trace_sources", fidelity_effect="trajectory truth preprocessing"),
    "experiment.subblocks.trace_source.processing.filter.method": _entry(valid_values=_values(("bessel", "implemented", "Bessel high-pass filter.")), consumed_by="campaign_trace_sources", fidelity_effect="trajectory truth preprocessing"),
    "experiment.subblocks.trace_source.processing.filter.apply_stage": _entry(valid_values=_values(("before_window", "implemented", "Apply filtering before subblock window selection."), ("after_window", "implemented", "Apply filtering after window selection.")), consumed_by="campaign_trace_sources", fidelity_effect="trajectory truth preprocessing"),
    "experiment.subblocks.trace_jitter.apply_to": _entry(valid_values=_values(("downstream_template_override", "implemented", "Forward as subblock-study trace-template jitter override; not added to materialized trajectory CSV."), ("trajectory_residual", "future_placeholder", "Future additive jitter on top of trajectory residual."), ("disabled", "implemented", "No trace jitter override.")), consumed_by="wrapper/subblock runner"),
    "experiment.iterative.update_mode": _entry(valid_values=_values(("physical_full", "implemented", "Physical basis full update."), ("eigen_full", "implemented", "Full eigenbasis projection equivalence update."), ("eigen_damped", "implemented", "Damped eigenbasis update."), ("eigen_truncated", "implemented", "Truncated eigenbasis update.")), consumed_by="observation_bias"),
    "experiment.iterative.eigenbasis.basis_source": _entry(valid_values=_values(("posterior_precision", "implemented", "Build modes from prior plus accumulated information."), ("accumulated_information", "implemented", "Build modes from measurement-only accumulated information.")), consumed_by="observation_bias update policy"),
    "experiment.iterative.eigenbasis.gate_source": _entry(valid_values=_values(("posterior_precision", "implemented", "Gate modes using prior plus accumulated information."), ("accumulated_information", "implemented", "Gate modes using measurement-only accumulated information.")), consumed_by="observation_bias update policy"),
    "experiment.iterative.eigenbasis.damping_mode": _entry(valid_values=_values(("none", "implemented", "No eigen-coordinate damping."), ("scalar", "implemented", "Use one scalar damping factor for every mode."), ("information", "implemented", "Use lambda/(lambda+damping_value) damping."), ("bottom_n", "implemented", "Damp only the weakest N gate modes by damping_value.")), consumed_by="observation_bias update policy"),
    "experiment.iterative.update_safety.posterior_sigma_policy": _entry(valid_values=_values(("reported_only", "implemented", "Do not inflate posterior sigma; report diagnostics only."), ("inflate_by_factor", "implemented", "Use existing posterior_sigma_inflation factor."), ("floor_only", "future_placeholder", "Future sigma floors only."), ("process_noise_floor", "future_placeholder", "Future process-noise floor.")), consumed_by="observation_bias"),
    "experiment.seeding.seed_policy": _entry(valid_values=_values(("different_jitter_different_noise", "implemented", "Distinct jitter and noise seeds."), ("same_jitter_different_noise", "implemented", "Shared jitter, distinct noise."), ("different_jitter_same_noise", "implemented", "Distinct jitter, shared noise.")), consumed_by="seeding"),
    "experiment.observation_theta.optics.primary_zernikes.indices": _entry(valid_values=_values(("from_system", "implemented", "Activate all resolved primary Zernike vector entries.")), consumed_by="observation theta layout", notes="List[int] is also implemented; index 0 maps to first resolved coefficient, currently Noll Z4."),
    "experiment.observation_theta.optics.secondary_zernikes.indices": _entry(valid_values=_values(("from_system", "implemented", "Activate all resolved secondary Zernike vector entries.")), consumed_by="observation theta layout", notes="List[int] is also implemented; index 0 maps to first resolved coefficient, currently Noll Z4."),
    "experiment.prior_draws.center": _entry(valid_values=_values(("truth", "implemented", "Draw around truth values.")), consumed_by="observation_bias"),
    "experiment.prior_draws.distribution": _entry(valid_values=_values(("normal", "implemented", "Independent Gaussian draws.")), consumed_by="observation_bias"),
    "experiment.prior_draws.sigmas.*.kind": _entry(valid_values=_values(("absolute", "implemented", "Sigma is in native parameter units."), ("fractional", "implemented", "Sigma is fraction of current truth/reference value."), ("ppm", "future_placeholder", "Future parts-per-million shorthand."), ("percent", "future_placeholder", "Future percent shorthand.")), consumed_by="observation_bias"),
    "experiment.prior_draws.sigmas.*.unit": _entry(consumed_by="observation_bias/provenance", provenance_effect="native unit label for prior draw sigma"),
    "experiment.truth_realization.enabled": _entry(consumed_by="observation_bias", fidelity_effect="optional truth scalar overrides"),
    "experiment.eigenbasis.enabled": _entry(implemented_status="accepted_but_noop", consumed_by="observation_bias", notes="Disabled in current executable configs."),
    "experiment.forecast.enabled": _entry(implemented_status="accepted_but_noop", consumed_by="observation_bias", notes="Disabled in current executable configs."),
}

# Common numeric/bool/list fields without enum validation but with documented effects.
for _path, _meta in {
    "experiment.seed": ("wrapper/seeding", "reproducible realization"),
    "experiment.spectral_model.truth.n_lambda": ("spectral resolver", "truth wavelength count"),
    "experiment.spectral_model.truth.wavelength_min_nm": ("spectral resolver", "truth band minimum"),
    "experiment.spectral_model.truth.wavelength_max_nm": ("spectral resolver", "truth band maximum"),
    "experiment.spectral_model.inference.n_lambda": ("spectral resolver", "reference wavelength count"),
    "experiment.spectral_model.inference.wavelength_min_nm": ("spectral resolver", "reference band minimum"),
    "experiment.spectral_model.inference.wavelength_max_nm": ("spectral resolver", "reference band maximum"),
    "experiment.high_order_wfe.truth.npix": ("campaign_high_order_wfe", "WFE map sampling"),
    "experiment.high_order_wfe.truth.amplitude_nm_rms": ("campaign_high_order_wfe", "truth WFE RMS"),
    "experiment.high_order_wfe.truth.power_law_alpha": ("campaign_high_order_wfe", "truth WFE spectrum"),
    "experiment.high_order_wfe.inference.knowledge_error.amplitude_nm_rms": ("campaign_high_order_wfe", "error WFE RMS"),
    "experiment.subblocks.n_subblocks": ("observation_bias", "subblock count"),
    "experiment.subblocks.n_frames": ("subblock runner", "frame count"),
    "experiment.subblocks.exposure_time_s": ("system resolver/subblock runner", "source exposure/frame cadence"),
    "experiment.subblocks.reference_n_iter": ("subblock runner", "optimizer iterations"),
    "experiment.subblocks.reference_base_lr": ("subblock runner", "optimizer learning rate"),
    "experiment.subblocks.reference_early_stopping_min_iter": ("subblock runner", "early stopping"),
    "experiment.subblocks.reference_early_stopping_patience": ("subblock runner", "early stopping"),
    "experiment.subblocks.noise.variance_floor": ("subblock runner", "canonical variance model floor"),
    "experiment.subblocks.variance_floor": ("subblock runner", "deprecated variance floor alias"),
    "experiment.iterative.update_gain": ("observation_bias", "update damping"),
    "experiment.iterative.windows_per_draw": ("observation_bias", "iterative windows"),
    "experiment.iterative.subblocks_per_window": ("observation_bias", "subblocks per update"),
    "experiment.iterative.eigenbasis.whiten": ("observation_bias update policy", "prior-whitened eigenbasis selection"),
    "experiment.iterative.eigenbasis.eig_floor_abs": ("observation_bias update policy", "absolute mode gate"),
    "experiment.iterative.eigenbasis.eig_floor_rel": ("observation_bias update policy", "relative mode gate"),
    "experiment.iterative.eigenbasis.damping_value": ("observation_bias update policy", "eigen-coordinate damping strength"),
    "experiment.iterative.eigenbasis.damping_n_modes": ("observation_bias update policy", "number of weakest gate modes to damp"),
    "experiment.iterative.eigenbasis.min_kept_modes": ("observation_bias update policy", "minimum retained eigenmodes"),
    "experiment.iterative.eigenbasis.max_kept_modes": ("observation_bias update policy", "maximum retained eigenmodes"),
    "experiment.iterative.eigenbasis.top_k_contributors": ("observation_bias update policy", "mode contributor diagnostic count"),
    "experiment.iterative.update_safety.posterior_sigma_inflation": ("observation_bias", "sigma inflation when policy=inflate_by_factor"),
    "experiment.prior_draws.case_name_template": ("observation_bias", "case naming"),
}.items():
    CONFIG_FIELD_REGISTRY.setdefault(_path, _entry(consumed_by=_meta[0], runtime_effect=_meta[1]))


def _pattern_to_regex(pattern: str) -> re.Pattern[str]:
    escaped = re.escape(pattern).replace(r"\*", r".+")
    return re.compile(f"^{escaped}$")


def registry_entry_for_path(path: str) -> tuple[str, dict[str, Any]] | tuple[None, None]:
    if path in CONFIG_FIELD_REGISTRY:
        return path, CONFIG_FIELD_REGISTRY[path]
    for pattern, entry in CONFIG_FIELD_REGISTRY.items():
        if "*" in pattern and _pattern_to_regex(pattern).match(path):
            return pattern, entry
    return None, None


def flatten_leaf_paths(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        out: list[tuple[str, Any]] = []
        for key, child in value.items():
            out.extend(flatten_leaf_paths(child, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(value, list):
        return [(prefix, value)]
    return [(prefix, value)]


def iter_string_fields(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [(path, value) for path, value in flatten_leaf_paths(config) if isinstance(value, str)]


def validate_config_contract(config: Mapping[str, Any], *, config_tier: str = "review", strict: bool = False) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    for path, value in flatten_leaf_paths(config):
        pattern, entry = registry_entry_for_path(path)
        if isinstance(value, str):
            if entry is None:
                findings.append({"severity": "error" if strict else "warning", "field_path": path, "value": value, "code": "undocumented_string_field", "message": "String-valued field has no registry entry."})
                continue
            valid = entry.get("valid_values", {})
            if valid and value not in valid:
                # Numeric-as-string values are not expected for enum fields.
                findings.append({"severity": "error", "field_path": path, "value": value, "code": "unsupported_enum_value", "message": f"Unsupported value {value!r}; valid values: {', '.join(valid)}."})
            elif valid and valid.get(value, {}).get("status") == "future_placeholder":
                findings.append({"severity": "error" if strict else "warning", "field_path": path, "value": value, "code": "future_value_used", "message": f"Value {value!r} is documented as future/deferred, not implemented."})
            elif valid and valid.get(value, {}).get("status") == "smoke_only" and config_tier == "review":
                findings.append({"severity": "error" if strict else "warning", "field_path": path, "value": value, "code": "smoke_only_value_in_review", "message": "Smoke-only value used in review config."})
        if entry is not None and entry.get("implemented_status") == "smoke_only" and config_tier == "review":
            findings.append({"severity": "error" if strict else "warning", "field_path": path, "value": value, "code": "smoke_only_field_in_review", "message": "Smoke-only field is not allowed in review config."})
        if entry is not None and entry.get("implemented_status") == "future_placeholder":
            findings.append({"severity": "error" if strict else "warning", "field_path": path, "value": value, "code": "future_field_used", "message": "Field is documented as future/deferred."})
    if config_tier == "review" and _get(config, "experiment.spectral_model.fast") is not None:
        findings.append({"severity": "error", "field_path": "experiment.spectral_model.fast", "value": _get(config, "experiment.spectral_model.fast"), "code": "fast_in_review_config", "message": "spectral_model.fast is smoke-only and must be absent from review config."})
    return {"findings": findings, "has_errors": any(f["severity"] == "error" for f in findings)}


def _get(config: Mapping[str, Any], path: str) -> Any:
    cur: Any = config
    for part in path.split("."):
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(part)
    return cur


def registry_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, entry in sorted(CONFIG_FIELD_REGISTRY.items()):
        row = copy.deepcopy(entry)
        row["field_path"] = path
        row["valid_values"] = json.dumps(row.get("valid_values", {}), sort_keys=True)
        rows.append(row)
    return rows


def write_reference_docs(markdown_path: str | Path, json_path: str | Path) -> None:
    rows = registry_rows()
    Path(json_path).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    cols = ["field_path", "valid_values", "default", "implemented_status", "consumed_by", "runtime_effect", "fidelity_effect", "provenance_effect", "safe_to_omit", "notes"]
    lines = ["# Full-Fidelity Binary Iterative Review Config Reference", "", "Generated from `dluxshera.utils.full_fidelity_config_schema.CONFIG_FIELD_REGISTRY`.", "", "| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")).replace("|", "\\|").replace("\n", " ") for c in cols) + " |")
    Path(markdown_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "CONFIG_FIELD_REGISTRY",
    "IMPLEMENTED_STATUSES",
    "flatten_leaf_paths",
    "iter_string_fields",
    "registry_entry_for_path",
    "registry_rows",
    "validate_config_contract",
    "write_reference_docs",
]
