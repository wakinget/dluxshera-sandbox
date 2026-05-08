from __future__ import annotations

import csv
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "scripts"
    / "run_obs_subblock_study.py"
)
RECIPE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "recipes"
    / "observation_subblock_inference.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "run_obs_subblock_study_script",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_recipe_module():
    spec = importlib.util.spec_from_file_location(
        "observation_subblock_inference_recipe_unit_tests",
        RECIPE_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _trace_template() -> dict:
    return {
        "system": {
            "preset": "SHERA_TESTBED_3P",
            "source": {"target": "ALPHA_CEN", "exposure_time_s": 0.05},
            "optics": {"plate_scale_as_per_pix": 0.01},
        },
        "experiment": {
            "kind": "subblock_trace_generation",
            "trace": {
                "n_frames": 3,
                "dt_s": 0.05,
                "varying_keys": ["source.x_position_as"],
                "plan": {
                    "source.x_position_as": {
                        "base": 0.0,
                        "effects": [{"kind": "constant_offset", "offset": 0.0}],
                    }
                },
            },
            "outputs": {"outdir": "Results/default", "file_prefix": "subblock_trace"},
        },
    }


def _render_template() -> dict:
    return {
        "system": {
            "preset": "SHERA_TESTBED_3P",
            "source": {"exposure_time_s": 0.05},
            "optics": {"plate_scale_as_per_pix": 0.01},
        },
        "experiment": {
            "kind": "subblock_generation",
            "truth": {"optics": {"plate_scale_as_per_pix": 0.01}},
            "subblock": {
                "trace": {"format": "csv", "path": "placeholder_trace.csv"},
            },
            "noise": {"enabled": False},
            "outputs": {"outdir": "Results/default", "file_prefix": "obs_subblock"},
        },
    }


def _inference_template() -> dict:
    return {
        "system": {
            "preset": "SHERA_TESTBED_3P",
            "source": {"exposure_time_s": 0.05},
            "optics": {"plate_scale_as_per_pix": 0.01},
        },
        "experiment": {
            "kind": "subblock_inference",
            "inference": {
                "data": {
                    "cube": "placeholder_cube.fits",
                    "truth_trace": "placeholder_truth.csv",
                    "manifest": "placeholder_manifest.json",
                },
                "validate": {
                    "require_contiguous_frame_index": True,
                    "require_monotonic_time": True,
                },
                "active": {
                    "frame_keys": [
                        "source.x_position_as",
                        "source.y_position_as",
                        "source.position_angle_deg",
                    ],
                    "shared_keys": [],
                },
                "init": {
                    "frame": {
                        "mode": "shared_guess",
                        "values": {
                            "source.x_position_as": 0.0,
                            "source.y_position_as": 0.0,
                            "source.position_angle_deg": 0.0,
                        },
                    },
                    "shared": {},
                },
                "priors": {"frame": {}, "shared": {}},
                "temporal": {"frame_model": {"kind": "independent"}},
                "objective": {
                    "kind": "nll",
                    "frame_reduce": "sum",
                    "subblock_reduce": "sum",
                    "noise_model": {
                        "kind": "gaussian",
                        "variance_model": "data",
                        "variance_floor": 1.0,
                    },
                },
                "optimizer": {
                    "kind": "sgd",
                    "base_lr": 0.5,
                    "n_iter": 2,
                    "preconditioning": {"enabled": False},
                },
                "diagnostics": {
                    "plots": True,
                    "compare_to_truth_when_available": True,
                },
            },
            "outputs": {
                "outdir": "Results/default",
                "file_prefix": "subblock_inference",
            },
        },
    }


def _write_templates(tmp_path: Path) -> tuple[Path, Path, Path]:
    trace_path = tmp_path / "trace_template.json"
    render_path = tmp_path / "render_template.json"
    inference_path = tmp_path / "inference_template.json"
    _write_json(trace_path, _trace_template())
    _write_json(render_path, _render_template())
    _write_json(inference_path, _inference_template())
    return trace_path, render_path, inference_path


def test_fisher_noise_audit_uses_resolved_data_variance_floor():
    script = _load_script_module()
    recipe = _load_recipe_module()
    cube = np.array([[[0.0, 0.5, 1.0, 2.0]]], dtype=float)
    noise_model_cfg = {
        "kind": "gaussian",
        "variance_model": "data",
        "variance_floor": 0.5,
    }
    variance_cube = recipe._build_variance_cube(
        data_cube=cube,
        noise_model_cfg=noise_model_cfg,
    )

    audit = script._build_fisher_noise_audit(
        {
            "cube": cube,
            "variance_cube": variance_cube,
            "recipe": recipe,
            "inference_cfg": {"objective": {"noise_model": noise_model_cfg}},
            "manifest": None,
            "manifest_path": None,
        }
    )

    assert audit["data_variance_floor_value"] == 0.5
    assert audit["data_variance_floor_source"] == "explicit_config"
    assert audit["data_variance_floor_clipped_count"] == 2


def _write_case_render_artifacts(
    case_root: Path,
    *,
    truth_value: float = 0.01,
    shared_truth: dict | None = None,
    resolved_system: dict | None = None,
) -> tuple[Path, Path, Path, Path]:
    render_dir = case_root / "render"
    render_dir.mkdir(parents=True, exist_ok=True)
    cube_path = render_dir / "obs_subblock_cube.fits"
    cube_path.write_bytes(b"cube")
    variance_path = render_dir / "obs_subblock_variance.fits"
    variance_path.write_bytes(b"variance")
    truth_path = render_dir / "obs_subblock_truth.csv"
    truth_path.write_text(
        "frame_index,time_s,source.x_position_as,source.y_position_as\n"
        "0,0.0,0.0,0.0\n"
        "1,0.1,0.1,-0.1\n",
        encoding="utf-8",
    )
    manifest_path = render_dir / "manifest.json"
    manifest_shared_truth = (
        {"optics": {"plate_scale_as_per_pix": truth_value}}
        if shared_truth is None
        else shared_truth
    )
    manifest_resolved_system = (
        {"optics": {"plate_scale_as_per_pix": truth_value}}
        if resolved_system is None
        else resolved_system
    )
    _write_json(
        manifest_path,
        {
            "artifacts": {
                "cube_fits": cube_path.name,
                "variance_fits": variance_path.name,
                "frame_truth_csv": truth_path.name,
            },
            "shared_truth": manifest_shared_truth,
            "system": {"resolved_config": manifest_resolved_system},
        },
    )
    return cube_path, variance_path, truth_path, manifest_path


def _resolve_relative(config_path: Path, value: str) -> Path:
    return (config_path.parent / value).resolve()


@dataclass(frozen=True)
class _ResolvedInput:
    path: Path | None
    source: str | None = None


@dataclass(frozen=True)
class _RenderInputs:
    cube: _ResolvedInput
    truth_trace: _ResolvedInput
    manifest: _ResolvedInput


def test_parse_helpers_validate_mode_candidate_and_grid():
    module = _load_script_module()

    assert module.parse_study_mode("profile_objective") == "profile_objective"
    assert module.parse_study_mode("schur_summary") == "schur_summary"
    assert module.parse_theta_keys(None) == (
        "source.separation_as",
        "source.log_flux_total",
        "source.contrast",
        "optics.plate_scale_as_per_pix",
    )
    assert module.normalize_schur_phi_ref_mode("truth") == "truth_when_available"
    assert module.parse_scalar_candidate_parameter("optics.plate_scale_as_per_pix") == (
        "optics.plate_scale_as_per_pix"
    )
    assert module.parse_scalar_candidate_parameter(
        "optics.primary.zernike_coeffs_nm[3]"
    ) == "optics.primary.zernike_coeffs_nm[3]"
    assert module.parse_scalar_grid("0.99,1.0,1.01") == (0.99, 1.0, 1.01)

    with pytest.raises(ValueError, match="Unsupported study mode"):
        module.parse_study_mode("profile")

    with pytest.raises(ValueError, match="Invalid observation-subblock key syntax"):
        module.parse_scalar_candidate_parameter("optics.primary.zernike_coeffs_nm[abc]")

    with pytest.raises(ValueError, match="Unsupported observation-subblock varying key"):
        module.parse_scalar_candidate_parameter("detector.read_noise_e")

    with pytest.raises(ValueError, match="at least one value"):
        module.parse_scalar_grid("")


def test_trace_template_resolution_uses_schur_specific_default(tmp_path: Path):
    module = _load_script_module()

    schur_path, schur_source = module.resolve_study_trace_template(
        mode="schur_summary",
        trace_template=None,
    )
    fisher_path, fisher_source = module.resolve_study_trace_template(
        mode="fisher_only",
        trace_template=None,
    )
    override_path = tmp_path / "custom_trace.yaml"
    explicit_path, explicit_source = module.resolve_study_trace_template(
        mode="schur_summary",
        trace_template=override_path,
    )
    explicit_fisher_path, explicit_fisher_source = module.resolve_study_trace_template(
        mode="fisher_only",
        trace_template=override_path,
    )

    assert schur_path == module.DEFAULT_SCHUR_TRACE_TEMPLATE.resolve()
    assert schur_source == "schur_summary_default"
    assert fisher_path == module.DEFAULT_TRACE_TEMPLATE.resolve()
    assert fisher_source == "general_default"
    assert explicit_path == override_path.resolve()
    assert explicit_source == "cli_override"
    assert explicit_fisher_path == override_path.resolve()
    assert explicit_fisher_source == "cli_override"


def test_schur_workflow_policy_defaults_are_discoverable():
    module = _load_script_module()

    defaults = module.SCHUR_WORKFLOW_DEFAULTS
    reference_policy = module.SCHUR_REFERENCE_INFERENCE_POLICY

    assert defaults.trace_template == module.DEFAULT_SCHUR_TRACE_TEMPLATE
    assert defaults.theta_keys == module.DEFAULT_SCHUR_THETA_KEYS
    assert defaults.phi_ref == "truth_when_available"
    assert module.DEFAULT_SCHUR_MAX_DENSE_DIM == 40
    assert defaults.max_dense_dim == module.DEFAULT_SCHUR_MAX_DENSE_DIM
    assert defaults.validate_structured_against_dense is False
    assert reference_policy.preconditioning_enabled == module.TEMPLATE_OWNED_DEFAULT


def test_registration_iid_template_supports_trace_jitter_overrides():
    module = _load_script_module()
    cfg = module.load_config_file(module.DEFAULT_SCHUR_TRACE_TEMPLATE)

    applied = module._apply_trace_truth_overrides(
        cfg,
        truth_overrides={
            "trace_x0_as": 0.0,
            "trace_y0_as": 0.0,
            "trace_pa0_deg": 14.508,
        },
        jitter_overrides={
            "trace_jitter_x_sigma_as": 0.12,
            "trace_jitter_y_sigma_as": 0.12,
            "trace_jitter_pa_sigma_deg": 0.002,
        },
        seed=42,
    )
    plan = cfg["experiment"]["trace"]["plan"]

    assert plan["source.x_position_as"]["effects"][0]["sigma"] == pytest.approx(0.12)
    assert plan["source.y_position_as"]["effects"][0]["sigma"] == pytest.approx(0.12)
    assert plan["source.position_angle_deg"]["effects"][0]["sigma"] == pytest.approx(0.002)
    assert applied["jitter"]["source.position_angle_deg"]["effect_kind"] == "iid_jitter"


def test_registration_iid_template_has_no_hidden_plate_scale_offset():
    module = _load_script_module()
    cfg = module.load_config_file(module.DEFAULT_SCHUR_TRACE_TEMPLATE)
    trace_cfg = cfg["experiment"]["trace"]
    plan = trace_cfg["plan"]

    assert "optics.plate_scale_as_per_pix" not in trace_cfg.get("varying_keys", [])
    plate_entry = plan.get("optics.plate_scale_as_per_pix")
    if plate_entry is not None:
        effects = plate_entry.get("effects", [])
        assert not any(
            effect.get("kind") == "constant_offset"
            and float(effect.get("offset", 0.0)) == pytest.approx(0.0002)
            for effect in effects
        )


def test_schur_theta_key_validation_accepts_four_scalar_smoke_keys():
    module = _load_script_module()

    classification = module.validate_schur_summary_theta_keys(
        (
            "source.separation_as",
            "source.log_flux_total",
            "source.contrast",
            "optics.plate_scale_as_per_pix",
        )
    )
    assert classification["supported"] == [
        "source.separation_as",
        "source.log_flux_total",
        "source.contrast",
        "optics.plate_scale_as_per_pix",
    ]
    assert classification["blocked"] == []


def test_validate_schur_dense_dimension_fails_clearly():
    module = _load_script_module()

    with pytest.raises(ValueError, match="exceeds max_dense_dim=10"):
        module._validate_schur_dense_dimension(combined_dim=11, max_dense_dim=10)


def test_observation_theta_layout_zernike_toggle_is_explicit():
    module = _load_script_module()

    no_zernikes = module._build_observation_theta_layout(
        theta_keys=("source.separation_as",),
        enable_zernikes=False,
        zernike_indices=(0, 1),
    )
    with_zernikes = module._build_observation_theta_layout(
        theta_keys=("source.separation_as",),
        enable_zernikes=True,
        zernike_indices=(0, 1),
    )

    assert no_zernikes.labels == ("source.separation_as",)
    assert with_zernikes.labels == (
        "source.separation_as",
        "optics.primary.zernike_coeffs_nm[0]",
        "optics.primary.zernike_coeffs_nm[1]",
        "optics.secondary.zernike_coeffs_nm[0]",
        "optics.secondary.zernike_coeffs_nm[1]",
    )


def test_schur_theta_runtime_update_supports_grad_and_hessian_for_four_scalar_set(
    tmp_path: Path,
):
    module = _load_script_module()
    _trace_template, _render_template, inference_template = _write_templates(tmp_path)
    template_context = module._resolve_template_system_context(inference_template)
    theta_layout = module._build_observation_theta_layout(
        theta_keys=(
            "source.separation_as",
            "source.log_flux_total",
            "source.contrast",
            "optics.plate_scale_as_per_pix",
        ),
        enable_zernikes=False,
        zernike_indices=(0, 1),
    )
    theta_addresses = module._theta_addresses_for_layout(
        theta_layout=theta_layout,
        forward_spec=template_context["forward_spec"],
        base_store=template_context["store"],
    )
    theta_ref = module._observation_theta_ref_from_store(
        theta_layout=theta_layout,
        base_store=template_context["store"],
    )

    def _loss(theta_values):
        updated = module._apply_theta_overrides(
            reference_store=template_context["store"],
            forward_spec=template_context["forward_spec"],
            theta_addresses=theta_addresses,
            theta_values=module.jnp.asarray(theta_values, dtype=float),
        )
        raw_fluxes = module.jnp.asarray(updated.get("source.raw_fluxes"), dtype=float)
        return (
            0.1 * module.jnp.asarray(updated.get("source.separation_as"), dtype=float)
            + 1.0e-6 * raw_fluxes[0]
            + 2.0e-6 * raw_fluxes[1]
            + 10.0 * module.jnp.asarray(updated.get("optics.plate_scale_as_per_pix"), dtype=float)
        )

    grad = module.jax.grad(_loss)(module.jnp.asarray(theta_ref, dtype=float))
    hess = module.jax.hessian(_loss)(module.jnp.asarray(theta_ref, dtype=float))
    updated = module._apply_theta_overrides(
        reference_store=template_context["store"],
        forward_spec=template_context["forward_spec"],
        theta_addresses=theta_addresses,
        theta_values=module.jnp.asarray(theta_ref + np.array([0.0, 0.3, -0.2, 0.001]), dtype=float),
    )

    assert grad.shape == (4,)
    assert hess.shape == (4, 4)
    assert np.all(np.isfinite(np.asarray(grad)))
    assert np.all(np.isfinite(np.asarray(hess)))
    assert float(np.asarray(updated.get("source.log_flux_total"))) == pytest.approx(
        float(theta_ref[1] + 0.3)
    )


def test_derive_scalar_information_metrics_handles_nonpositive_marginal_info():
    module = _load_script_module()

    valid = module.derive_scalar_information_metrics(f_pp=4.0, i_marg=1.0)
    assert valid["sigma_cond"] == pytest.approx(0.5)
    assert valid["sigma_marg"] == pytest.approx(1.0)
    assert valid["absorption_fraction"] == pytest.approx(0.75)
    assert valid["marginalization_status"] == "ok"
    assert valid["valid_marginal_sigma"] is True

    zero_marg = module.derive_scalar_information_metrics(f_pp=4.0, i_marg=0.0)
    assert zero_marg["sigma_marg"] == float("inf")
    assert zero_marg["valid_marginal_sigma"] is False
    assert zero_marg["marginalization_status"] == "zero_marginal_information"

    negative_marg = module.derive_scalar_information_metrics(f_pp=4.0, i_marg=-1.0)
    assert negative_marg["sigma_marg"] is None
    assert negative_marg["valid_marginal_sigma"] is False
    assert negative_marg["absorption_fraction"] == pytest.approx(1.25)
    assert negative_marg["marginalization_status"] == "negative_marginal_information"


def test_finite_difference_information_from_cube_derivative_respects_reductions():
    module = _load_script_module()

    dmodel_dp = module.np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.5, 1.0], [1.5, 2.0]],
        ],
        dtype=float,
    )
    variance_cube = module.np.ones_like(dmodel_dp)

    info_sum = module._finite_difference_information_from_cube_derivative(
        dmodel_dp=dmodel_dp,
        variance_cube=variance_cube,
        frame_reduce="sum",
        subblock_reduce="sum",
    )
    info_mean = module._finite_difference_information_from_cube_derivative(
        dmodel_dp=dmodel_dp,
        variance_cube=variance_cube,
        frame_reduce="sum",
        subblock_reduce="mean",
    )

    assert info_sum == pytest.approx(37.5)
    assert info_mean == pytest.approx(18.75)


def test_classify_candidate_runtime_status_distinguishes_disconnected_candidates():
    module = _load_script_module()

    assert (
        module._classify_candidate_runtime_status(
            candidate_found_in_layout=True,
            field_found=True,
            binding_present=True,
            store_changed=True,
            model_changes=False,
            finite_difference_f_pp=0.0,
            fisher_f_pp=0.0,
        )
        == "candidate_changes_store_but_not_model"
    )
    assert (
        module._classify_candidate_runtime_status(
            candidate_found_in_layout=True,
            field_found=True,
            binding_present=True,
            store_changed=True,
            model_changes=True,
            finite_difference_f_pp=2.0,
            fisher_f_pp=0.0,
        )
        == "fisher_assembly_suspect"
    )


def test_preserve_shared_derived_active_values_restores_shared_candidate_after_frame_update():
    recipe = _load_recipe_module()
    candidate_key = "optics.plate_scale_as_per_pix"
    shared_spec = recipe.ActiveKeySpec(
        canonical=candidate_key,
        address=recipe.parse_obs_subblock_varying_keys([candidate_key])[0],
        kind="derived",
    )
    shared_store = recipe.ParameterStore.from_dict(
        {
            candidate_key: 0.12443914,
            "source.x_position_as": 0.0,
        }
    )
    frame_store = recipe.ParameterStore.from_dict(
        {
            candidate_key: 0.12320707,
            "source.x_position_as": 0.05,
        }
    )

    preserved = recipe._preserve_shared_derived_active_values(
        frame_store=frame_store,
        shared_store=shared_store,
        shared_specs=(shared_spec,),
    )

    assert float(preserved.get(candidate_key)) == pytest.approx(0.12443914)
    assert float(preserved.get("source.x_position_as")) == pytest.approx(0.05)


def test_fisher_only_auto_switches_to_structured_arrowhead_for_large_theta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_script_module()
    candidate_key = "optics.plate_scale_as_per_pix"
    n_frame = 11
    frame_width = 3
    shared_width = 1
    theta_size = n_frame * frame_width + shared_width

    class _FakeRecipe:
        class jnp:
            @staticmethod
            def asarray(value):
                return value

        @staticmethod
        def _unpack_active_state(layout, theta_flat):
            frame_size = layout.n_frame * layout.frame_width
            frame_flat = theta_flat[:frame_size]
            frame_rows = [
                frame_flat[index * layout.frame_width : (index + 1) * layout.frame_width]
                for index in range(layout.n_frame)
            ]
            return SimpleNamespace(frame=frame_rows, shared=theta_flat[frame_size:])

        @staticmethod
        def build_independent_frame_curvature_blocks(**_kwargs):
            blocks = tuple(
                SimpleNamespace(
                    frame_block=module.np.eye(frame_width) * 4.0,
                    coupling_block=module.np.full((frame_width, shared_width), 0.5),
                    shared_block=module.np.asarray([[2.0]]),
                )
                for _ in range(n_frame)
            )
            return SimpleNamespace(blocks=blocks, shared_dim=shared_width)

        @staticmethod
        def fim_theta(*_args, **_kwargs):
            raise AssertionError("dense fim_theta should not be used for theta_size > 30")

    fake_layout = SimpleNamespace(
        n_frame=n_frame,
        frame_width=frame_width,
        shared_width=shared_width,
        theta_size=theta_size,
        frame_keys=(
            "source.x_position_as",
            "source.y_position_as",
            "source.position_angle_deg",
        ),
        shared_keys=(candidate_key,),
    )
    theta_reference = module.np.zeros(theta_size, dtype=float)

    monkeypatch.setattr(
        module,
        "_prepare_inference_context",
        lambda **_kwargs: {
            "recipe": _FakeRecipe,
            "layout": fake_layout,
            "theta_reference": theta_reference,
            "theta_reference_source": "truth_trace",
            "objective_bundle": SimpleNamespace(frame_data_term_fn=lambda *_args: 0.0),
            "inference_cfg": {"objective": {"subblock_reduce": "sum"}},
            "system_cfg": {"source": {"target": "ALPHA_CEN"}},
        },
    )
    monkeypatch.setattr(
        module,
        "_evaluate_candidate_sensitivity",
        lambda **_kwargs: {
            "conclusion": "candidate_changes_model",
            "compact": {
                "candidate_model_rms_delta_1pct": 0.25,
                "candidate_loss_delta_1pct": 0.5,
                "frame_store_preserves_candidate": True,
                "finite_difference_f_pp": 21.5,
                "candidate_runtime_status": "candidate_changes_model",
            },
        },
    )
    monkeypatch.setattr(
        module,
        "_build_fisher_noise_audit",
        lambda _context: {
            "variance_model": "provided_cube",
            "variance_source": "provided_cube",
            "cube_stats": {"mean": 2.0},
            "variance_stats": {"mean": 3.0},
        },
    )

    summary = module._evaluate_fisher_only(
        config_path=tmp_path / "fisher_config.json",
        output_dir=tmp_path / "study",
        candidate_key=candidate_key,
        truth_value=0.01,
        noise_mode="noiseless",
        target_name="ALPHA_CEN",
    )

    assert summary["fisher_method"] == "structured_arrowhead"
    assert summary["dense_global_fim_materialized"] is False
    assert summary["structured_block_count"] == n_frame
    assert summary["f_pp"] == pytest.approx(22.0)
    assert summary["i_marg"] == pytest.approx(19.9375)
    assert summary["candidate_runtime_status"] == "candidate_changes_model"
    assert summary["frame_store_preserves_candidate"] is True
    assert summary["finite_difference_f_pp"] == pytest.approx(21.5)
    assert summary["noise_audit"]["variance_source"] == "provided_cube"


def test_schur_curvature_method_selection_prefers_dense_below_guard():
    module = _load_script_module()

    used = module._select_schur_curvature_method(
        requested_method="auto",
        combined_dim=12,
        max_dense_dim=20,
        structured_support={"supported": True, "unsupported_reasons": []},
    )

    assert used == "dense"


def test_schur_curvature_method_selection_uses_structured_above_guard():
    module = _load_script_module()

    used = module._select_schur_curvature_method(
        requested_method="auto",
        combined_dim=64,
        max_dense_dim=60,
        structured_support={"supported": True, "unsupported_reasons": []},
    )

    assert used == "structured_independent_frames"


def test_schur_curvature_method_selection_uses_structured_for_20_frame_default_guard():
    module = _load_script_module()

    used = module._select_schur_curvature_method(
        requested_method="auto",
        combined_dim=64,
        max_dense_dim=module.DEFAULT_SCHUR_MAX_DENSE_DIM,
        structured_support={"supported": True, "unsupported_reasons": []},
    )

    assert used == "structured_independent_frames"


def test_dense_vs_structured_comparison_state_is_opt_in_and_guarded():
    module = _load_script_module()

    default_state = module._dense_vs_structured_comparison_state(
        requested=False,
        curvature_method_used="structured_independent_frames",
        combined_dim=34,
        max_dense_dim=40,
    )
    assert default_state["dense_vs_structured_comparison_run"] is False
    assert default_state["dense_vs_structured_comparison_skipped_reason"] == "not_requested"

    run_state = module._dense_vs_structured_comparison_state(
        requested=True,
        curvature_method_used="structured_independent_frames",
        combined_dim=34,
        max_dense_dim=40,
    )
    assert run_state["dense_vs_structured_comparison_run"] is True

    skipped_state = module._dense_vs_structured_comparison_state(
        requested=True,
        curvature_method_used="structured_independent_frames",
        combined_dim=64,
        max_dense_dim=40,
    )
    assert skipped_state["dense_vs_structured_comparison_run"] is False
    assert (
        skipped_state["dense_vs_structured_comparison_skipped_reason"]
        == "combined_dim_exceeds_max_dense_dim"
    )


def test_schur_curvature_method_selection_rejects_unsupported_layout():
    module = _load_script_module()

    with pytest.raises(ValueError, match="shared active subblock state"):
        module._select_schur_curvature_method(
            requested_method="structured_independent_frames",
            combined_dim=64,
            max_dense_dim=80,
            structured_support={
                "supported": False,
                "unsupported_reasons": ["shared active subblock state is configured"],
            },
        )


def test_fisher_only_dry_run_writes_shared_candidate_config(tmp_path: Path):
    module = _load_script_module()
    trace_template, render_template, inference_template = _write_templates(tmp_path)
    case_root = tmp_path / "Results" / "case_fisher"
    cube_path, variance_path, truth_path, manifest_path = _write_case_render_artifacts(
        case_root,
        truth_value=0.0125,
    )

    summary = module.run_obs_subblock_study(
        mode="fisher_only",
        case_root=case_root,
        trace_template=trace_template,
        render_template=render_template,
        inference_template=inference_template,
        candidate_key="optics.plate_scale_as_per_pix",
        truth_value=0.0125,
        use_render_variance=True,
        dry_run=True,
    )

    summary_path = case_root / "study" / "fisher_only" / "summary.json"
    fisher_config_path = case_root / "study" / "fisher_only" / "fisher" / "inference_config.json"
    fisher_cfg = _read_json(fisher_config_path)

    assert summary["summary_path"] == str(summary_path.resolve())
    assert summary["case_prep_stages_executed"] == []
    assert summary["rendered_truth_value"] == 0.0125
    assert summary["candidate_base_key"] == "optics.plate_scale_as_per_pix"
    assert summary["candidate_index"] is None
    assert summary["target_name"] == "ALPHA_CEN"
    assert fisher_cfg["experiment"]["inference"]["active"]["shared_keys"] == [
        "optics.plate_scale_as_per_pix"
    ]
    assert fisher_cfg["experiment"]["inference"]["init"]["shared"] == {
        "optics.plate_scale_as_per_pix": 0.0125
    }
    assert _resolve_relative(
        fisher_config_path,
        fisher_cfg["experiment"]["inference"]["data"]["cube"],
    ) == cube_path.resolve()
    assert _resolve_relative(
        fisher_config_path,
        fisher_cfg["experiment"]["inference"]["data"]["truth_trace"],
    ) == truth_path.resolve()
    assert _resolve_relative(
        fisher_config_path,
        fisher_cfg["experiment"]["inference"]["data"]["manifest"],
    ) == manifest_path.resolve()
    assert (
        fisher_cfg["experiment"]["inference"]["objective"]["noise_model"]["variance_model"]
        == "provided_cube"
    )
    assert _resolve_relative(
        fisher_config_path,
        fisher_cfg["experiment"]["inference"]["objective"]["noise_model"]["path"],
    ) == variance_path.resolve()


def test_candidate_address_validation_rejects_out_of_range_index(tmp_path: Path):
    module = _load_script_module()
    _trace_template, _render_template, inference_template = _write_templates(tmp_path)
    template_context = module._resolve_template_system_context(inference_template)

    with pytest.raises(ValueError, match="out of bounds"):
        module.parse_candidate_parameter_address(
            "optics.primary.zernike_coeffs_nm[999]",
            forward_spec=template_context["forward_spec"],
            reference_store=template_context["store"],
        )


def test_fisher_only_dry_run_supports_indexed_candidate_metadata_and_init(tmp_path: Path):
    module = _load_script_module()
    trace_template, render_template, inference_template = _write_templates(tmp_path)
    case_root = tmp_path / "Results" / "case_fisher_indexed"
    cube_path, variance_path, truth_path, manifest_path = _write_case_render_artifacts(
        case_root,
        shared_truth={
            "optics": {
                "primary": {"zernike_coeffs_nm": [0.0, 0.0, 0.0, 12.5, 0.0, 0.0, 0.0, 0.0]}
            }
        },
        resolved_system={
            "optics": {
                "primary": {"zernike_coeffs_nm": [0.0, 0.0, 0.0, 12.5, 0.0, 0.0, 0.0, 0.0]}
            }
        },
    )

    summary = module.run_obs_subblock_study(
        mode="fisher_only",
        case_root=case_root,
        trace_template=trace_template,
        render_template=render_template,
        inference_template=inference_template,
        candidate_key="optics.primary.zernike_coeffs_nm[3]",
        truth_value=12.5,
        use_render_variance=True,
        dry_run=True,
    )

    fisher_config_path = case_root / "study" / "fisher_only" / "fisher" / "inference_config.json"
    fisher_cfg = _read_json(fisher_config_path)
    render_cfg = _read_json(case_root / "study" / "fisher_only" / "templates" / "render_template.json")
    trace_cfg = _read_json(case_root / "study" / "fisher_only" / "templates" / "trace_template.json")

    assert summary["candidate_parameter"] == "optics.primary.zernike_coeffs_nm[3]"
    assert summary["candidate_base_key"] == "optics.primary.zernike_coeffs_nm"
    assert summary["candidate_index"] == 3
    assert summary["rendered_truth_value"] == pytest.approx(12.5)
    assert summary["case_prep_stages_executed"] == []
    assert fisher_cfg["experiment"]["inference"]["active"]["shared_keys"] == [
        "optics.primary.zernike_coeffs_nm[3]"
    ]
    assert fisher_cfg["experiment"]["inference"]["init"]["shared"] == {
        "optics.primary.zernike_coeffs_nm[3]": 12.5
    }
    assert (
        render_cfg["experiment"]["truth"]["optics"]["primary"]["zernike_coeffs_nm"][3]
        == pytest.approx(12.5)
    )
    assert (
        trace_cfg["system"]["optics"]["primary"]["zernike_coeffs_nm"][3]
        == pytest.approx(12.5)
    )
    assert _resolve_relative(
        fisher_config_path,
        fisher_cfg["experiment"]["inference"]["data"]["cube"],
    ) == cube_path.resolve()
    assert _resolve_relative(
        fisher_config_path,
        fisher_cfg["experiment"]["inference"]["data"]["truth_trace"],
    ) == truth_path.resolve()
    assert _resolve_relative(
        fisher_config_path,
        fisher_cfg["experiment"]["inference"]["data"]["manifest"],
    ) == manifest_path.resolve()
    assert _resolve_relative(
        fisher_config_path,
        fisher_cfg["experiment"]["inference"]["objective"]["noise_model"]["path"],
    ) == variance_path.resolve()


def test_fisher_only_dry_run_without_existing_render_is_plan_only(tmp_path: Path):
    module = _load_script_module()
    trace_template, render_template, inference_template = _write_templates(tmp_path)
    case_root = tmp_path / "Results" / "case_fisher_plan_only"

    summary = module.run_obs_subblock_study(
        mode="fisher_only",
        case_root=case_root,
        trace_template=trace_template,
        render_template=render_template,
        inference_template=inference_template,
        candidate_key="optics.plate_scale_as_per_pix",
        truth_value=0.0125,
        dry_run=True,
    )

    assert summary["dry_run"] is True
    assert summary["case_prep_stages_executed"] == ["trace", "render"]
    assert "case_prep_summary_path" not in summary


def test_profile_objective_reuses_existing_render_outputs_and_writes_curve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_script_module()
    trace_template, render_template, inference_template = _write_templates(tmp_path)
    case_root = tmp_path / "Results" / "case_profile"
    cube_path, _variance_path, _truth_path, _manifest_path = _write_case_render_artifacts(case_root)

    def fake_inference_runner(config_path: Path, run_root: Path, dry_run: bool) -> dict:
        assert dry_run is False
        cfg = _read_json(config_path)
        assumed_value = float(cfg["system"]["optics"]["plate_scale_as_per_pix"])
        output_dir = run_root / "inference"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "manifest.json").write_text("{}", encoding="utf-8")
        return {
            "initial_loss": 10.0,
            "final_loss": abs(assumed_value - 1.0),
            "output_dir": str(output_dir),
            "artifacts": {
                "manifest_json": str((output_dir / "manifest.json").resolve()),
            },
        }

    monkeypatch.setattr(module, "_default_inference_runner", fake_inference_runner)

    summary = module.run_obs_subblock_study(
        mode="profile_objective",
        case_root=case_root,
        trace_template=trace_template,
        render_template=render_template,
        inference_template=inference_template,
        candidate_key="optics.plate_scale_as_per_pix",
        scan_values=(0.99, 1.0, 1.01),
        dry_run=False,
    )

    profile_summary = summary["profile_summary"]
    curve_path = Path(profile_summary["curve_csv"])
    rows = list(csv.DictReader(curve_path.open("r", encoding="utf-8", newline="")))
    first_config_path = Path(rows[0]["config_path"])
    first_cfg = _read_json(first_config_path)

    assert summary["case_prep_stages_executed"] == []
    assert profile_summary["best_run"]["assumed_value"] == 1.0
    assert profile_summary["summary_path"] == str(
        (case_root / "study" / "profile_objective" / "summary.json").resolve()
    )
    assert len(rows) == 3
    assert [float(row["assumed_value"]) for row in rows] == [0.99, 1.0, 1.01]
    assert _resolve_relative(
        first_config_path,
        first_cfg["experiment"]["inference"]["data"]["cube"],
    ) == cube_path.resolve()


def test_nuisance_absorption_writes_bias_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_script_module()
    trace_template, render_template, inference_template = _write_templates(tmp_path)
    case_root = tmp_path / "Results" / "case_absorption"
    _write_case_render_artifacts(case_root, truth_value=0.011)

    def fake_inference_runner(config_path: Path, run_root: Path, dry_run: bool) -> dict:
        assert dry_run is False
        output_dir = run_root / "inference"
        output_dir.mkdir(parents=True, exist_ok=True)
        truth_csv = output_dir / "truth_comparison.csv"
        with truth_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "source.x_position_as_residual",
                    "source.y_position_as_residual",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "source.x_position_as_residual": "1.0",
                    "source.y_position_as_residual": "-1.0",
                }
            )
            writer.writerow(
                {
                    "source.x_position_as_residual": "3.0",
                    "source.y_position_as_residual": "-3.0",
                }
            )
        return {
            "initial_loss": 6.0,
            "final_loss": 2.0,
            "output_dir": str(output_dir),
            "frame_keys": [
                "source.x_position_as",
                "source.y_position_as",
            ],
            "artifacts": {
                "truth_comparison_csv": str(truth_csv.resolve()),
            },
        }

    monkeypatch.setattr(module, "_default_inference_runner", fake_inference_runner)

    summary = module.run_obs_subblock_study(
        mode="nuisance_absorption",
        case_root=case_root,
        trace_template=trace_template,
        render_template=render_template,
        inference_template=inference_template,
        candidate_key="optics.plate_scale_as_per_pix",
        assumed_value=0.009,
        dry_run=False,
    )

    bias_summary = summary["nuisance_absorption_summary"]["bias_summary"]
    assert summary["rendered_truth_value"] == 0.011
    assert bias_summary["frame_count"] == 2
    assert bias_summary["per_key"]["source.x_position_as"]["mean_bias"] == 2.0
    assert bias_summary["per_key"]["source.y_position_as"]["max_abs_residual"] == 3.0
    assert bias_summary["overall"]["rms_residual"] == pytest.approx(
        (5.0**0.5),
        rel=1.0e-9,
    )


def test_profile_objective_errors_when_rendered_cube_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_script_module()
    trace_template, render_template, inference_template = _write_templates(tmp_path)
    case_root = tmp_path / "Results" / "case_missing_cube"

    def fake_prepare_case_render_artifacts(**_kwargs) -> dict:
        return {
            "render_inputs": _RenderInputs(
                cube=_ResolvedInput(None),
                truth_trace=_ResolvedInput(None),
                manifest=_ResolvedInput(None),
            ),
            "case_prep_summary": None,
            "stages_executed": [],
        }

    monkeypatch.setattr(
        module,
        "_prepare_case_render_artifacts",
        fake_prepare_case_render_artifacts,
    )

    with pytest.raises(ValueError, match="requires a rendered cube"):
        module.run_obs_subblock_study(
            mode="profile_objective",
            case_root=case_root,
            trace_template=trace_template,
            render_template=render_template,
            inference_template=inference_template,
            candidate_key="optics.plate_scale_as_per_pix",
            scan_values=(1.0,),
            dry_run=False,
        )


def test_full_case_mode_delegates_to_case_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_script_module()
    trace_template, render_template, inference_template = _write_templates(tmp_path)
    case_root = tmp_path / "Results" / "case_full"
    recorded: dict[str, object] = {}

    class _FakeCaseModule:
        @staticmethod
        def run_case_workflow(**kwargs) -> dict:
            recorded.update(kwargs)
            return {
                "summary_path": str((case_root / "case_summary.json").resolve()),
                "stages_requested": ["trace", "render"],
            }

    monkeypatch.setattr(module, "_load_case_runner_module", lambda: _FakeCaseModule)

    summary = module.run_obs_subblock_study(
        mode="full_case",
        case_root=case_root,
        case_stages="trace,render",
        trace_template=trace_template,
        render_template=render_template,
        inference_template=inference_template,
        dry_run=True,
    )

    assert recorded["case_root"] == case_root
    assert recorded["stages"] == "trace,render"
    assert summary["case_summary_path"] == str((case_root / "case_summary.json").resolve())
    assert summary["case_stages_requested"] == ["trace", "render"]


def test_schur_summary_dry_run_writes_config_and_planned_artifacts(tmp_path: Path):
    module = _load_script_module()
    trace_template, render_template, inference_template = _write_templates(tmp_path)
    case_root = tmp_path / "Results" / "case_schur_dry_run"
    _write_case_render_artifacts(case_root, truth_value=0.011)

    summary = module.run_obs_subblock_study(
        mode="schur_summary",
        case_root=case_root,
        trace_template=trace_template,
        render_template=render_template,
        inference_template=inference_template,
        theta_keys=("source.separation_as", "optics.plate_scale_as_per_pix"),
        phi_ref="truth_when_available",
        dry_run=True,
    )

    assert summary["dry_run"] is True
    assert "schur_config_path" in summary
    assert Path(summary["schur_config_path"]).exists()
    assert "planned_artifacts" in summary
    assert "schur_summary_plan_path" in summary
    plan_path = Path(summary["schur_summary_plan_path"])
    assert plan_path.exists()
    plan = _read_json(plan_path)
    assert plan["n_frames"] == 3
    assert plan["theta_labels"] == [
        "source.separation_as",
        "optics.plate_scale_as_per_pix",
    ]
    assert plan["phi_ref_mode"] == "truth_when_available"
    assert plan["combined_dim"] == 11
    assert plan["max_dense_dim"] == module.DEFAULT_SCHUR_MAX_DENSE_DIM
    assert plan["dense_hessian_allowed"] is True
    assert plan["dense_vs_structured_comparison_requested"] is False
    assert plan["dense_vs_structured_comparison_run"] is False
    assert (
        plan["dense_vs_structured_comparison_skipped_reason"]
        == "curvature_method_not_structured"
    )
    assert plan["planned_artifacts"]["subblock_summary_json"].endswith(
        "subblock_summary.json"
    )
    assert "trace_truth" in plan
    assert "inference_init" in plan
    assert plan["reference_inference_will_run"] is False
    assert plan["preconditioning_actually_used"] is False
    assert plan["preconditioning_not_used_reason"] == "reference inference did not run"
    assert plan["summary_export_inference_config_path"].endswith("inference_config.json")
    assert plan["reference_inference_config_if_run"]["sources"][
        "preconditioning_enabled"
    ] == "inference_template"
    assert plan["preconditioning"]["sources"]["preconditioning_enabled"] == "inference_template"
    assert "diagnostics" in plan["reference_inference_config_if_run"]
    summary_export_cfg = _read_json(Path(plan["summary_export_inference_config_path"]))
    summary_noise_model = summary_export_cfg["experiment"]["inference"]["objective"][
        "noise_model"
    ]
    assert summary_noise_model["variance_model"] == "data"
    assert summary_noise_model["variance_floor"] == 1.0
    assert plan["planned_artifacts"]["frame_truth_preview_json"].endswith(
        "frame_truth_preview.json"
    )
    assert plan["planned_artifacts"]["schur_summary_audit_json"].endswith(
        "schur_summary_audit.json"
    )
    assert (case_root / "study" / "schur_summary" / "summary.json").exists()


def test_build_schur_summary_plan_records_recovered_unpreconditioned_warning(tmp_path: Path):
    module = _load_script_module()
    trace_template, render_template, inference_template = _write_templates(tmp_path)
    case_root = tmp_path / "Results" / "case_schur_plan_warning"
    _write_case_render_artifacts(case_root, truth_value=0.011)
    render_inputs = _RenderInputs(
        cube=_ResolvedInput(case_root / "render" / "obs_subblock_cube.fits"),
        truth_trace=_ResolvedInput(case_root / "render" / "obs_subblock_truth.csv"),
        manifest=_ResolvedInput(case_root / "render" / "manifest.json"),
    )
    summary_run_root = case_root / "study" / "schur_summary" / "summary_export"
    schur_cfg = module._build_study_inference_config(
        template_path=inference_template,
        run_root=summary_run_root,
        render_inputs=render_inputs,
        exposure_time_s=None,
        candidate_key=None,
        assumed_value=None,
        force_truth_comparison=False,
        disable_plots=True,
        use_render_variance=False,
    )
    schur_config_path = summary_run_root / "inference_config.json"
    _write_json(schur_config_path, schur_cfg)

    plan = module._build_schur_summary_plan(
        case_root=case_root,
        study_root=case_root / "study" / "schur_summary",
        template_paths={
            "trace": trace_template,
            "render": render_template,
            "inference": inference_template,
        },
        source_template_paths={
            "trace": trace_template,
            "render": render_template,
            "inference": inference_template,
        },
        trace_template_source="cli_override",
        schur_config_path=schur_config_path,
        schur_config=schur_cfg,
        schur_config_provenance=module._build_schur_config_provenance(
            schur_config=schur_cfg,
            reference_preconditioning_enabled=None,
            reference_preconditioning_reference=None,
            reference_diagnostics_profile=None,
            force_truth_comparison=False,
        ),
        render_inputs=render_inputs,
        case_prep_stages=[],
        n_frames_requested=3,
        dt_s_requested=None,
        exposure_time_s_requested=None,
        noise_mode="disabled",
        theta_keys=("source.separation_as", "optics.plate_scale_as_per_pix"),
        enable_zernikes=False,
        zernike_indices=(0, 1),
        schur_damping=1.0e-8,
        max_dense_dim=80,
        schur_curvature_method="auto",
        phi_ref_mode="recovered",
        summary_objective="full_objective",
        validate_surrogate=True,
        validation_steps=5,
        frame_truth_preview=None,
        applied_trace_overrides={},
        applied_inference_init_overrides={},
    )

    assert plan["reference_inference_will_run"] is True
    assert plan["reference_inference_config_if_run"]["optimizer_kind"] == "sgd"
    assert plan["reference_inference_config_if_run"]["preconditioning_enabled"] is False
    assert plan["preconditioning_actually_used"] is False
    assert any(
        "phi_ref=recovered will use unpreconditioned SGD" in warning
        for warning in plan["known_limitations_or_warnings"]
    )


def test_schur_summary_recovered_plan_reports_preconditioning_enabled_cli_source(
    tmp_path: Path,
):
    module = _load_script_module()
    _trace_template, render_template, inference_template = _write_templates(tmp_path)
    case_root = tmp_path / "Results" / "case_schur_recovered_precond"
    _write_case_render_artifacts(case_root, truth_value=0.011)

    summary = module.run_obs_subblock_study(
        mode="schur_summary",
        case_root=case_root,
        render_template=render_template,
        inference_template=inference_template,
        theta_keys=("source.separation_as", "optics.plate_scale_as_per_pix"),
        phi_ref="recovered",
        reference_preconditioning_enabled=True,
        reference_preconditioning_reference="initial",
        reference_diagnostics_profile="review",
        dry_run=True,
    )

    plan = _read_json(Path(summary["schur_summary_plan_path"]))
    reference = plan["reference_inference_config_if_run"]
    assert plan["reference_inference_will_run"] is True
    assert reference["preconditioning_enabled"] is True
    assert plan["preconditioning_actually_used"] is True
    assert plan["preconditioning_not_used_reason"] is None
    assert reference["sources"]["preconditioning_enabled"] == "cli_override"
    assert reference["sources"]["preconditioning_reference"] == "cli_override"
    assert reference["preconditioning_reference"] == "initial"
    assert reference["diagnostics"]["settings"]["first_step_report"] is True
    assert reference["diagnostics"]["sources"]["first_step_report"] == "cli_override"

    generated_cfg = _read_json(Path(summary["schur_config_path"]))
    preconditioning = generated_cfg["experiment"]["inference"]["optimizer"][
        "preconditioning"
    ]
    assert preconditioning["enabled"] is True
    assert preconditioning["reference"] == "initial"


def test_schur_summary_recovered_plan_records_reference_optimizer_overrides(
    tmp_path: Path,
):
    module = _load_script_module()
    _trace_template, render_template, inference_template = _write_templates(tmp_path)
    case_root = tmp_path / "Results" / "case_schur_recovered_optimizer"
    _write_case_render_artifacts(case_root, truth_value=0.011)

    summary = module.run_obs_subblock_study(
        mode="schur_summary",
        case_root=case_root,
        render_template=render_template,
        inference_template=inference_template,
        theta_keys=("source.separation_as", "optics.plate_scale_as_per_pix"),
        phi_ref="recovered",
        reference_optimizer_kind="adam",
        reference_base_lr=1.0e-3,
        reference_n_iter=300,
        reference_optimizer_kwargs={"b1": "0.8", "b2": "0.999", "eps": "1e-8"},
        reference_preconditioning_enabled=True,
        reference_preconditioning_method="dense",
        reference_preconditioning_reference="truth_when_available",
        reference_preconditioning_damping=1.0e-5,
        reference_preconditioning_eig_floor_rel=1.0e-6,
        reference_preconditioning_eig_floor_abs=1.0e-8,
        reference_preconditioning_lr_clip=(0.1, 10.0),
        dry_run=True,
    )

    generated_cfg = _read_json(Path(summary["schur_config_path"]))
    optimizer = generated_cfg["experiment"]["inference"]["optimizer"]
    assert optimizer["kind"] == "adam"
    assert optimizer["base_lr"] == pytest.approx(1.0e-3)
    assert optimizer["n_iter"] == 300
    assert optimizer["kwargs"] == {"b1": 0.8, "b2": 0.999, "eps": 1.0e-8}
    preconditioning = optimizer["preconditioning"]
    assert preconditioning["enabled"] is True
    assert preconditioning["method"] == "dense_full_theta"
    assert preconditioning["reference"] == "truth_when_available"
    assert preconditioning["damping"] == pytest.approx(1.0e-5)
    assert preconditioning["eig_floor_rel"] == pytest.approx(1.0e-6)
    assert preconditioning["eig_floor_abs"] == pytest.approx(1.0e-8)
    assert preconditioning["lr_clip"] == [0.1, 10.0]

    plan = _read_json(Path(summary["schur_summary_plan_path"]))
    reference = plan["reference_inference_config_if_run"]
    assert reference["optimizer_kind"] == "adam"
    assert reference["base_lr"] == pytest.approx(1.0e-3)
    assert reference["n_iter"] == 300
    assert reference["optimizer_kwargs"] == {"b1": 0.8, "b2": 0.999, "eps": 1.0e-8}
    assert reference["preconditioning_method"] == "dense_full_theta"
    assert reference["sources"]["optimizer_kind"] == "cli_override"
    assert reference["sources"]["base_lr"] == "cli_override"
    assert reference["sources"]["n_iter"] == "cli_override"
    assert reference["sources"]["optimizer_kwargs"] == "cli_override"
    assert reference["sources"]["preconditioning_method"] == "cli_override"
    assert reference["sources"]["preconditioning_lr_clip"] == "cli_override"


def test_reference_optimizer_override_validation_errors():
    module = _load_script_module()
    cfg = {"optimizer": {"kind": "sgd", "base_lr": 0.5, "n_iter": 2}}
    with pytest.raises(ValueError, match="reference_base_lr"):
        module.apply_reference_optimizer_overrides(cfg, base_lr=0.0)
    with pytest.raises(ValueError, match="reference_n_iter"):
        module.apply_reference_optimizer_overrides(cfg, n_iter=0)
    with pytest.raises(ValueError, match="not supported for optimizer kind"):
        module.apply_reference_optimizer_overrides(
            cfg,
            optimizer_kind="adam",
            optimizer_kwargs={"typo": "1.0"},
        )


def test_reference_optimizer_parser_accepts_recovered_controls():
    module = _load_script_module()
    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "schur_summary",
            "--case-root",
            "/tmp/case",
            "--reference-optimizer-kind",
            "adam",
            "--reference-base-lr",
            "1e-3",
            "--reference-n-iter",
            "300",
            "--reference-optimizer-kwarg",
            "b1=0.8",
            "--reference-preconditioning-enabled",
            "--reference-preconditioning-method",
            "auto",
            "--reference-preconditioning-reference",
            "initial",
            "--reference-preconditioning-damping",
            "1e-6",
            "--reference-preconditioning-eig-floor-rel",
            "1e-6",
            "--reference-preconditioning-eig-floor-abs",
            "1e-8",
            "--reference-preconditioning-lr-clip",
            "0.1,10",
            "--validate-structured-against-dense",
        ]
    )
    assert args.reference_optimizer_kind == "adam"
    assert args.reference_base_lr == pytest.approx(1.0e-3)
    assert args.reference_n_iter == 300
    assert args.reference_optimizer_kwarg == ["b1=0.8"]
    assert args.reference_preconditioning_enabled is True
    assert args.reference_preconditioning_method == "auto"
    assert args.reference_preconditioning_reference == "initial"
    assert args.validate_structured_against_dense is True
    assert module.parse_reference_preconditioning_lr_clip(
        args.reference_preconditioning_lr_clip
    ) == (0.1, 10.0)


def test_memory_snapshot_payload_is_json_serializable():
    module = _load_script_module()
    payload = module.capture_memory_snapshot("unit.stage", n_frames=5)
    assert payload["stage"] == "unit.stage"
    assert payload["pid"] > 0
    json.dumps(payload)


def test_memory_diagnostics_writer_appends_jsonl_records(tmp_path: Path):
    module = _load_script_module()
    path = tmp_path / "memory.jsonl"
    recorder = module.MemoryDiagnosticsRecorder(path)
    recorder.record("stage.one", arrays=module.named_array_memory_metadata(x=np.zeros((2, 3))))
    recorder.record("stage.two", dtype="float64")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [record["stage"] for record in records] == ["stage.one", "stage.two"]
    assert records[0]["metadata"]["arrays"]["x"]["shape"] == [2, 3]
    assert records[0]["metadata"]["arrays"]["x"]["dtype"] == "float64"
    assert records[0]["metadata"]["arrays"]["x"]["nbytes"] == 48
    audit = recorder.audit_payload(summary_json_written=False, matrix_npz_written=False)
    assert audit["last_stage"] == "stage.two"


def test_memory_diagnostics_cli_flags_parse():
    module = _load_script_module()
    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "schur_summary",
            "--case-root",
            "/tmp/case",
            "--memory-diagnostics",
            "--memory-diagnostics-file",
            "/tmp/memory.jsonl",
        ]
    )
    assert args.memory_diagnostics is True
    assert args.memory_diagnostics_file == Path("/tmp/memory.jsonl")


def test_schur_summary_recovered_plan_reports_preconditioning_disabled_template_source(
    tmp_path: Path,
):
    module = _load_script_module()
    _trace_template, render_template, inference_template = _write_templates(tmp_path)
    case_root = tmp_path / "Results" / "case_schur_recovered_no_precond"
    _write_case_render_artifacts(case_root, truth_value=0.011)

    summary = module.run_obs_subblock_study(
        mode="schur_summary",
        case_root=case_root,
        render_template=render_template,
        inference_template=inference_template,
        theta_keys=("source.separation_as", "optics.plate_scale_as_per_pix"),
        phi_ref="recovered",
        dry_run=True,
    )

    plan = _read_json(Path(summary["schur_summary_plan_path"]))
    reference = plan["reference_inference_config_if_run"]
    assert plan["reference_inference_will_run"] is True
    assert reference["preconditioning_enabled"] is False
    assert plan["preconditioning_actually_used"] is False
    assert plan["preconditioning_not_used_reason"] == (
        "preconditioning disabled in inference config"
    )
    assert reference["sources"]["preconditioning_enabled"] == "inference_template"


def test_schur_summary_dry_run_trace_and_init_overrides_are_recorded(tmp_path: Path):
    module = _load_script_module()
    trace_template, render_template, inference_template = _write_templates(tmp_path)
    case_root = tmp_path / "Results" / "case_schur_overrides"

    summary = module.run_obs_subblock_study(
        mode="schur_summary",
        case_root=case_root,
        trace_template=trace_template,
        render_template=render_template,
        inference_template=inference_template,
        theta_keys=("source.separation_as", "optics.plate_scale_as_per_pix"),
        phi_ref="truth_when_available",
        trace_x0_as=1.25,
        trace_seed=123,
        init_x_as=0.5,
        init_y_as=-0.25,
        init_pa_deg=14.508,
        dry_run=True,
    )

    copied_trace = _read_json(Path(summary["templates"]["trace"]))
    copied_inference = _read_json(Path(summary["templates"]["inference"]))
    assert copied_trace["experiment"]["seed"] == 123
    assert copied_trace["experiment"]["trace"]["plan"]["source.x_position_as"]["base"] == pytest.approx(1.25)
    init_values = copied_inference["experiment"]["inference"]["init"]["frame"]["values"]
    assert init_values["source.x_position_as"] == pytest.approx(0.5)
    assert init_values["source.y_position_as"] == pytest.approx(-0.25)
    assert init_values["source.position_angle_deg"] == pytest.approx(14.508)

    plan = _read_json(Path(summary["schur_summary_plan_path"]))
    assert plan["trace_truth"]["nominal_or_base_values"]["source.x_position_as"] == pytest.approx(1.25)
    assert plan["trace_truth"]["seed"] == 123
    assert plan["inference_init"]["first_frame_initial_values"]["source.x_position_as"] == pytest.approx(0.5)
    assert plan["inference_init"]["init_value_sources"]["source.x_position_as"] == "cli_override"


def test_schur_summary_dry_run_records_default_trace_template_source(tmp_path: Path):
    module = _load_script_module()
    _trace_template, render_template, inference_template = _write_templates(tmp_path)
    case_root = tmp_path / "Results" / "case_schur_default_trace_template"

    summary = module.run_obs_subblock_study(
        mode="schur_summary",
        case_root=case_root,
        render_template=render_template,
        inference_template=inference_template,
        theta_keys=("source.separation_as", "optics.plate_scale_as_per_pix"),
        phi_ref="truth_when_available",
        trace_x0_as=0.0,
        trace_y0_as=0.0,
        trace_pa0_deg=14.508,
        trace_jitter_x_sigma_as=0.12,
        trace_jitter_y_sigma_as=0.12,
        trace_jitter_pa_sigma_deg=0.002,
        dry_run=True,
    )

    plan = _read_json(Path(summary["schur_summary_plan_path"]))
    assert summary["source_templates"]["trace_source"] == "schur_summary_default"
    assert Path(summary["source_templates"]["trace"]) == module.DEFAULT_SCHUR_TRACE_TEMPLATE.resolve()
    assert plan["trace_template_source"] == "schur_summary_default"
    assert Path(plan["trace_template_path"]) == module.DEFAULT_SCHUR_TRACE_TEMPLATE.resolve()
    assert plan["registration_iid_trace_template_used"] is True
    assert plan["trace_truth"]["registration_iid_template_used"] is True
    assert plan["trace_truth"]["jitter_amplitudes"]["source.position_angle_deg"] == pytest.approx(0.002)


def test_trace_override_fails_clearly_when_template_lacks_key(tmp_path: Path):
    module = _load_script_module()
    trace_template, render_template, inference_template = _write_templates(tmp_path)

    with pytest.raises(ValueError, match="source.y_position_as"):
        module.run_obs_subblock_study(
            mode="schur_summary",
            case_root=tmp_path / "Results" / "case_schur_bad_override",
            trace_template=trace_template,
            render_template=render_template,
            inference_template=inference_template,
            theta_keys=("source.separation_as", "optics.plate_scale_as_per_pix"),
            trace_y0_as=-1.0,
            dry_run=True,
        )


def test_frame_truth_preview_generation_on_tiny_csv(tmp_path: Path):
    module = _load_script_module()
    csv_path = tmp_path / "truth.csv"
    csv_path.write_text(
        "frame_index,time_s,source.x_position_as,source.y_position_as,source.position_angle_deg\n"
        "0,0.0,1.0,-1.0,90.0\n"
        "1,0.1,1.2,-0.8,89.5\n",
        encoding="utf-8",
    )
    preview_path = tmp_path / "frame_truth_preview.json"

    preview = module._write_frame_truth_preview(
        trace_csv_path=csv_path,
        preview_path=preview_path,
    )

    assert preview_path.exists()
    assert preview["row_count"] == 2
    assert preview["first_rows"][0]["source.x_position_as"] == pytest.approx(1.0)
    assert preview["column_stats"]["source.position_angle_deg"]["median"] == pytest.approx(89.75)


def test_schur_audit_links_plan_summary_diagnostics_and_validation(tmp_path: Path):
    module = _load_script_module()
    study_root = tmp_path / "study" / "schur_summary"
    plan_path = study_root / "schur_summary_plan.json"
    validation_path = study_root / "local_surrogate_validation.csv"
    module._write_rows_csv(
        validation_path,
        [
            {
                "label": "source.separation_as",
                "predicted_delta": 2.0,
                "actual_delta_fixed_phi": 1.0,
            }
        ],
    )
    plan = {
        "case_name": "case",
        "selected_stages": ["schur_summary_export"],
        "reference_inference_will_run": False,
        "reference_inference_not_run_reason": "phi_ref_mode=truth_when_available",
        "final_reference_inference_config_path": str(
            (study_root / "summary_export" / "inference_config.json").resolve()
        ),
        "preconditioning": {"preconditioning_actually_used": False},
        "reference_inference_config_if_run": {
            "optimizer_kind": "sgd",
            "diagnostics": {
                "settings": {"plots": True},
                "sources": {"plots": "inference_template"},
            },
        },
        "phi_ref_mode": "truth_when_available",
        "n_phi": 3,
        "phi_labels": ["phi.frame[0].source.x_position_as"],
        "theta_labels": ["source.separation_as"],
        "trace_truth": {"n_frames": 1},
        "trace_template_path": str(module.DEFAULT_SCHUR_TRACE_TEMPLATE.resolve()),
        "trace_template_source": "schur_summary_default",
        "registration_iid_trace_template_used": True,
        "trace_config_path": str((study_root / "templates" / "trace_template.json").resolve()),
        "generated_case_trace_config_path": str((tmp_path / "trace_config.json").resolve()),
        "cube_path": "cube.fits",
        "render_config_path": "render_config.json",
        "render_noise_mode": "disabled",
        "inference_init": {"packed_theta_size": 3},
        "planned_artifacts": {
            "schur_summary_audit_json": str((study_root / "schur_summary_audit.json").resolve()),
            "frame_truth_preview_json": str((study_root / "frame_truth_preview.json").resolve()),
            "schur_diagnostics_json": str((study_root / "schur_diagnostics.json").resolve()),
            "local_surrogate_validation_csv": str(validation_path.resolve()),
        },
    }
    summary_payload = {
        "phi_ref_source": "truth_trace",
        "theta_ref": [1.0],
        "artifacts": {
            "subblock_summary_json": str((study_root / "subblock_summary.json").resolve()),
            "schur_diagnostics_json": str((study_root / "schur_diagnostics.json").resolve()),
            "local_surrogate_validation_csv": str(validation_path.resolve()),
        },
    }

    audit = module._build_schur_summary_audit(
        plan=plan,
        plan_path=plan_path,
        summary_payload=summary_payload,
        recovered_reference_metadata={},
        frame_truth_preview={"row_count": 1},
    )

    assert audit["plan_json"].endswith("schur_summary_plan.json")
    assert audit["actual_artifacts"]["subblock_summary_json"].endswith("subblock_summary.json")
    assert audit["trace_template"]["trace_template_source"] == "schur_summary_default"
    assert audit["trace_template"]["registration_iid_trace_template_used"] is True
    assert audit["reference_inference"]["final_generated_config_path"].endswith(
        "inference_config.json"
    )
    assert audit["reference_diagnostics"]["settings"]["plots"] is True
    assert audit["schur_summary_diagnostics_path"].endswith("schur_diagnostics.json")
    assert audit["local_surrogate_validation"]["labels_validated"] == ["source.separation_as"]
    assert audit["observation_prior_recommendation"]["prior_mean_source"] == "summary_theta_ref"


def test_evaluate_schur_summary_writes_required_artifacts_with_tiny_quadratic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_script_module()

    class _FakeStore:
        def __init__(self, values):
            self._values = dict(values)

        def get(self, key):
            return self._values[key]

    class _FakeRecipe:
        jnp = np

        @staticmethod
        def _theta_labels_for_layout(layout):
            labels = []
            for frame_index in range(layout.n_frame):
                labels.extend(
                    f"frame[{frame_index}].{key}" for key in layout.frame_keys
                )
            labels.extend(f"shared.{key}" for key in layout.shared_keys)
            return labels

    fake_layout = SimpleNamespace(
        n_frame=2,
        frame_width=2,
        shared_width=0,
        theta_size=4,
        frame_keys=("source.x_position_as", "source.y_position_as"),
        shared_keys=(),
    )
    fake_forward_spec = {
        "source.separation_as": SimpleNamespace(kind="primitive", structural=False),
        "optics.plate_scale_as_per_pix": SimpleNamespace(
            kind="derived",
            structural=False,
        ),
    }
    fake_store = _FakeStore(
        {
            "source.separation_as": 1.5,
            "source.exposure_time_s": 0.05,
            "optics.plate_scale_as_per_pix": 0.01,
        }
    )

    monkeypatch.setattr(
        module,
        "_prepare_inference_context",
        lambda **_kwargs: {
            "recipe": _FakeRecipe,
            "layout": fake_layout,
            "base_store": fake_store,
            "forward_spec": fake_forward_spec,
            "theta0": np.array([0.1, -0.2, 0.0, 0.0]),
            "initial_state": SimpleNamespace(),
            "truth": SimpleNamespace(),
            "cube_path": tmp_path / "cube.fits",
            "manifest_path": tmp_path / "manifest.json",
            "trace_path": tmp_path / "truth.csv",
            "system_cfg": {"preset": "TEST", "source": {"target": "ALPHA_CEN"}},
                "inference_cfg": {
                    "active": {
                        "frame_keys": [
                            "source.x_position_as",
                            "source.y_position_as",
                        ],
                        "shared_keys": [],
                    },
                    "priors": {"frame": {}, "shared": {}},
                "temporal": {"frame_model": {"kind": "independent"}},
                "objective": {"frame_reduce": "sum", "subblock_reduce": "sum"},
            },
        },
    )

    hessian = np.array(
        [
            [5.0, 0.2, 1.0, 0.0, 0.2, -0.1],
            [0.2, 4.0, 0.0, 0.5, -0.2, 0.1],
            [1.0, 0.0, 3.0, 0.1, 0.0, 0.0],
            [0.0, 0.5, 0.1, 2.5, 0.0, 0.0],
            [0.2, -0.2, 0.0, 0.0, 2.0, 0.3],
            [-0.1, 0.1, 0.0, 0.0, 0.3, 1.8],
        ],
        dtype=float,
    )
    hessian = 0.5 * (hessian + hessian.T)
    gradient = np.array([-1.5, 0.75, 0.25, -0.1, 0.3, -0.2], dtype=float)
    reference = np.array([1.5, 0.01, 0.1, -0.2, 0.0, 0.0], dtype=float)

    def _fake_build_combined_local_objective(**_kwargs):
        def _loss(local_vector):
            delta = module.jnp.asarray(local_vector, dtype=float) - module.jnp.asarray(
                reference,
                dtype=float,
            )
            return (
                module.jnp.asarray(gradient, dtype=float) @ delta
                + 0.5
                * delta
                @ module.jnp.asarray(hessian, dtype=float)
                @ delta
            )

        return _loss, {
            "objective_kind_requested": "data_only",
            "objective_kind_used": "data_only",
            "priors_nonempty": False,
            "temporal_kind": "independent",
        }

    monkeypatch.setattr(
        module,
        "_build_combined_local_objective",
        _fake_build_combined_local_objective,
    )

    summary = module._evaluate_schur_summary(
        config_path=tmp_path / "inference_config.json",
        output_dir=tmp_path / "study" / "schur_summary",
        case_root=tmp_path / "case",
        theta_keys=("source.separation_as", "optics.plate_scale_as_per_pix"),
        enable_zernikes=False,
        zernike_indices=(0, 1),
        schur_damping=1.0e-8,
        max_dense_dim=20,
        schur_curvature_method="dense",
        phi_ref="init",
        summary_objective="data_only",
        validate_surrogate=True,
        validation_steps=5,
    )

    artifacts = {
        name: Path(path)
        for name, path in summary["artifacts"].items()
        if path is not None
    }
    for key in (
        "subblock_summary_json",
        "subblock_summary_matrices_npz",
        "schur_diagnostics_json",
        "combined_curvature_diagnostics_json",
        "local_surrogate_validation_csv",
    ):
        assert key in artifacts
        assert artifacts[key].exists()
        assert artifacts[key].stat().st_size > 0

    payload = _read_json(artifacts["subblock_summary_json"])
    assert payload["theta_labels"] == [
        "source.separation_as",
        "optics.plate_scale_as_per_pix",
    ]
    assert payload["phi_labels"] == [
        "phi.frame[0].source.x_position_as",
        "phi.frame[0].source.y_position_as",
        "phi.frame[1].source.x_position_as",
        "phi.frame[1].source.y_position_as",
    ]
    assert payload["prior_context"]["recommended_prior_mean_source"] == "summary_theta_ref"
    assert payload["prior_context"]["effective_store_values"]["source.exposure_time_s"] == pytest.approx(0.05)
    assert summary["loaded_summary_theta_labels"] == payload["theta_labels"]


def test_evaluate_schur_summary_uses_structured_path_with_tiny_quadratic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_script_module()

    class _FakeStore:
        def __init__(self, values):
            self._values = dict(values)

        def get(self, key):
            return self._values[key]

    class _FakeRecipe:
        jnp = np

        @staticmethod
        def _theta_labels_for_layout(layout):
            labels = []
            for frame_index in range(layout.n_frame):
                labels.extend(
                    f"frame[{frame_index}].{key}" for key in layout.frame_keys
                )
            return labels

        @staticmethod
        def _unpack_active_state(layout, theta_flat):
            frame = np.asarray(theta_flat, dtype=float).reshape(
                layout.n_frame,
                layout.frame_width,
            )
            return SimpleNamespace(frame=frame, shared=np.asarray([], dtype=float))

    fake_layout = SimpleNamespace(
        n_frame=2,
        frame_width=2,
        shared_width=0,
        theta_size=4,
        frame_keys=("source.x_position_as", "source.y_position_as"),
        shared_keys=(),
    )
    fake_forward_spec = {
        "source.separation_as": SimpleNamespace(kind="primitive", structural=False),
        "optics.plate_scale_as_per_pix": SimpleNamespace(
            kind="derived",
            structural=False,
        ),
    }
    fake_store = _FakeStore(
        {
            "source.separation_as": 1.5,
            "source.exposure_time_s": 0.05,
            "optics.plate_scale_as_per_pix": 0.01,
        }
    )
    monkeypatch.setattr(
        module,
        "_prepare_inference_context",
        lambda **_kwargs: {
            "recipe": _FakeRecipe,
            "layout": fake_layout,
            "base_store": fake_store,
            "forward_spec": fake_forward_spec,
            "theta0": np.array([0.1, -0.2, 0.0, 0.0]),
            "initial_state": SimpleNamespace(),
            "truth": SimpleNamespace(),
            "cube_path": tmp_path / "cube.fits",
            "manifest_path": tmp_path / "manifest.json",
            "trace_path": tmp_path / "truth.csv",
            "system_cfg": {"preset": "TEST", "source": {"target": "ALPHA_CEN"}},
            "objective_bundle": SimpleNamespace(frame_data_term_fn=lambda *_args: 0.0),
            "inference_cfg": {
                "active": {
                    "frame_keys": [
                        "source.x_position_as",
                        "source.y_position_as",
                    ],
                    "shared_keys": [],
                },
                "priors": {"frame": {}, "shared": {}},
                "temporal": {"frame_model": {"kind": "independent"}},
                "objective": {"frame_reduce": "sum", "subblock_reduce": "sum"},
            },
        },
    )
    monkeypatch.setattr(
        module,
        "_build_combined_local_objective",
        lambda **_kwargs: (
            lambda local_vector: module.jnp.asarray(0.0)
            + 0.0 * module.jnp.sum(module.jnp.asarray(local_vector)),
            {
                "objective_kind_requested": "data_only",
                "objective_kind_used": "data_only",
                "priors_nonempty": False,
                "temporal_kind": "independent",
            },
        ),
    )

    theta_ref = np.array([1.5, 0.01], dtype=float)
    frame_phi_ref = np.array([[0.1, -0.2], [0.0, 0.0]], dtype=float)
    local_hessians = (
        np.array(
            [
                [5.0, 0.2, 0.5, -0.1],
                [0.2, 4.0, 0.25, 0.1],
                [0.5, 0.25, 3.0, 0.2],
                [-0.1, 0.1, 0.2, 2.5],
            ],
            dtype=float,
        ),
        np.array(
            [
                [4.0, -0.1, 0.2, 0.3],
                [-0.1, 3.5, -0.2, 0.1],
                [0.2, -0.2, 2.0, 0.1],
                [0.3, 0.1, 0.1, 1.8],
            ],
            dtype=float,
        ),
    )
    local_gradients = (
        np.array([-0.3, 0.2, 0.1, -0.1], dtype=float),
        np.array([0.2, -0.1, 0.05, 0.15], dtype=float),
    )

    def _fake_structured_frame_objective(**_kwargs):
        def _frame_loss(theta_values, phi_values, frame_index):
            local = module.jnp.concatenate((theta_values, phi_values), axis=0)
            ref = module.jnp.concatenate(
                (
                    module.jnp.asarray(theta_ref),
                    module.jnp.asarray(frame_phi_ref[frame_index]),
                ),
                axis=0,
            )
            delta = local - ref
            hessian = module.jnp.asarray(local_hessians[frame_index])
            gradient = module.jnp.asarray(local_gradients[frame_index])
            return gradient @ delta + 0.5 * delta @ hessian @ delta

        return _frame_loss, {
            "objective_kind_requested": "data_only",
            "objective_kind_used": "data_only",
            "priors_nonempty": False,
            "temporal_kind": "independent",
            "inference_objective": {"subblock_reduce": "sum"},
        }

    monkeypatch.setattr(
        module,
        "_build_structured_schur_frame_objective",
        _fake_structured_frame_objective,
    )

    summary = module._evaluate_schur_summary(
        config_path=tmp_path / "inference_config.json",
        output_dir=tmp_path / "study" / "schur_summary",
        case_root=tmp_path / "case",
        theta_keys=("source.separation_as", "optics.plate_scale_as_per_pix"),
        enable_zernikes=False,
        zernike_indices=(0, 1),
        schur_damping=1.0e-8,
        max_dense_dim=4,
        schur_curvature_method="structured_independent_frames",
        phi_ref="init",
        summary_objective="data_only",
        validate_surrogate=False,
        validation_steps=5,
    )

    artifacts = {
        name: Path(path)
        for name, path in summary["artifacts"].items()
        if path is not None
    }
    payload = _read_json(artifacts["subblock_summary_json"])
    diagnostics = _read_json(artifacts["schur_diagnostics_json"])

    assert summary["schur_curvature_method_used"] == "structured_independent_frames"
    assert summary["dense_global_hessian_materialized"] is False
    assert summary["structured_curvature_used"] is True
    assert summary["dense_vs_structured_comparison_requested"] is False
    assert summary["dense_vs_structured_comparison_run"] is False
    assert summary["dense_vs_structured_comparison_skipped_reason"] == "not_requested"
    assert payload["metadata"]["curvature"]["structured_curvature_used"] is True
    assert payload["metadata"]["curvature"]["dense_vs_structured_comparison_run"] is False
    assert diagnostics["structured_curvature_used"] is True
    loaded = module.load_subblock_summary(artifacts["subblock_summary_json"])
    assert loaded.theta_labels == (
        "source.separation_as",
        "optics.plate_scale_as_per_pix",
    )

    comparison_summary = module._evaluate_schur_summary(
        config_path=tmp_path / "inference_config.json",
        output_dir=tmp_path / "study" / "schur_summary_compare",
        case_root=tmp_path / "case",
        theta_keys=("source.separation_as", "optics.plate_scale_as_per_pix"),
        enable_zernikes=False,
        zernike_indices=(0, 1),
        schur_damping=1.0e-8,
        max_dense_dim=20,
        schur_curvature_method="structured_independent_frames",
        phi_ref="init",
        summary_objective="data_only",
        validate_surrogate=False,
        validate_structured_against_dense=True,
        validation_steps=5,
    )
    assert comparison_summary["dense_vs_structured_comparison_requested"] is True
    assert comparison_summary["dense_vs_structured_comparison_run"] is True
    assert comparison_summary["dense_vs_structured_comparison"] is not None
