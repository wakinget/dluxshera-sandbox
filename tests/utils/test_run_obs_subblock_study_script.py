from __future__ import annotations

import csv
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

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
                    "noise_model": {"kind": "gaussian", "variance_model": "data"},
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
