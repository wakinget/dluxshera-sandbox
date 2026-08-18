"""Tests for temporal registration comparison helpers."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "examples" / "scripts" / "run_temporal_registration_comparison.py"
    spec = importlib.util.spec_from_file_location("temporal_registration_comparison", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parser_default_max_dense_dim():
    module = _load_module()
    result = module.main(
        [
            "--results-root",
            "/tmp/unused",
            "--run-name",
            "noop",
            "--dry-run",
            "--case-filter",
            "drift_truth__independent_fit",
        ]
    )
    manifest = json.loads((Path(result["run_root"]) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_frames"] == 20
    plan = json.loads((Path(result["run_root"]) / "comparison_plan.json").read_text(encoding="utf-8"))
    assert len(plan) == 1


def _args_for_optimizer(**overrides):
    module = _load_module()
    parser = module.argparse.ArgumentParser()
    parser.add_argument("--optimizer-kind", default="sgd")
    parser.add_argument("--optimizer-base-lr", type=float, default=0.9)
    parser.add_argument("--optimizer-n-iter", type=int, default=None)
    parser.add_argument("--noiseless-optimizer-n-iter", type=int, default=200)
    parser.add_argument("--shotnoise-optimizer-n-iter", type=int, default=100)
    parser.add_argument("--schedule-kind", default="linear_warmup")
    parser.add_argument("--schedule-warmup-steps", type=int, default=10)
    parser.add_argument("--schedule-start-factor", type=float, default=0.125)
    parser.add_argument("--disable-schedule", action="store_true")
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return module, args


def test_optimizer_defaults_noiseless_and_shotnoise():
    module, args = _args_for_optimizer()
    case = module.CaseSpec(case_name="c", truth_model="linear_drift", fit_model="independent")
    noiseless = module._optimizer_settings_for_case(case, n_frames=3, noise_mode="disabled", args=args)
    shot = module._optimizer_settings_for_case(case, n_frames=20, noise_mode="enabled", args=args)
    assert noiseless.kind == "sgd"
    assert noiseless.base_lr == pytest.approx(0.9)
    assert noiseless.n_iter == 200
    assert shot.n_iter == 100
    assert noiseless.schedule == {"kind": "linear_warmup", "warmup_steps": 10, "start_factor": 0.125}


def test_optimizer_disable_schedule_and_overrides():
    module, args = _args_for_optimizer(
        disable_schedule=True,
        optimizer_n_iter=77,
        optimizer_base_lr=0.3,
        schedule_kind="linear_warmup",
        schedule_warmup_steps=3,
        schedule_start_factor=0.5,
    )
    case = module.CaseSpec(case_name="c", truth_model="linear_drift", fit_model="linear_drift_residual_jitter_prior")
    settings = module._optimizer_settings_for_case(case, n_frames=3, noise_mode="enabled", args=args)
    assert settings.base_lr == pytest.approx(0.3)
    assert settings.n_iter == 77
    assert settings.schedule is None


def test_schur_settings_per_case_routing():
    module = _load_module()
    independent = module.CaseSpec(case_name="a", truth_model="linear_drift", fit_model="independent")
    linear = module.CaseSpec(case_name="b", truth_model="linear_drift", fit_model="linear_drift")
    residual = module.CaseSpec(case_name="c", truth_model="linear_drift_residual_jitter", fit_model="linear_drift_residual_jitter_prior")
    ind = module._schur_settings_for_case(independent, requested_max_dense_dim=80)
    lin = module._schur_settings_for_case(linear, requested_max_dense_dim=80)
    res = module._schur_settings_for_case(residual, requested_max_dense_dim=40)
    assert ind["schur_curvature_method"] == "auto"
    assert ind["max_dense_dim"] == 40
    assert lin["schur_curvature_method"] == "structured_linear_drift"
    assert lin["max_dense_dim"] == 80
    assert res["schur_curvature_method"] == "structured_residual_prior"
    assert res["max_dense_dim"] == 40


def test_iid_trace_has_expected_columns_and_shape(tmp_path: Path):
    module = _load_module()
    rows, manifest = module.build_iid_registration_trace(
        n_frames=4,
        exposure_time_s=0.05,
        seed=123,
    )
    path = tmp_path / "frame_truth.csv"
    module.write_temporal_trace_csv(path, rows)

    with path.open("r", encoding="utf-8", newline="") as handle:
        loaded = list(csv.DictReader(handle))

    assert len(loaded) == 4
    assert tuple(loaded[0].keys()) == module.TRACE_COLUMNS
    assert manifest["truth_model"] == "iid_registration"
    assert manifest["frame_count"] == 4


def test_linear_drift_trace_is_exactly_linear_without_residual():
    module = _load_module()
    rows, manifest = module.build_linear_drift_trace(
        n_frames=5,
        exposure_time_s=0.1,
        seed=123,
        x_rate_as_per_s=1.25,
        y_rate_as_per_s=-0.5,
        pa_rate_deg_per_s=0.01,
    )
    values = np.asarray([[float(row[key]) for key in module.TRACE_KEYS] for row in rows])
    times = np.asarray([float(row["time_s"]) for row in rows])
    centered = times - np.mean(times)
    expected = np.asarray([0.0, 0.0, 90.0])[None, :] + centered[:, None] * np.asarray(
        [1.25, -0.5, 0.01]
    )[None, :]

    np.testing.assert_allclose(values, expected)
    assert manifest["trace_statistics"]["source.x_position_as"]["linear_residual_rms"] < 1e-14


def test_drift_residual_trace_is_reproducible():
    module = _load_module()
    first, first_manifest = module.build_linear_drift_residual_jitter_trace(
        n_frames=6,
        exposure_time_s=0.05,
        seed=55,
        residual_x_sigma_as=0.01,
        residual_y_sigma_as=0.01,
        residual_pa_sigma_deg=1e-4,
    )
    second, second_manifest = module.build_linear_drift_residual_jitter_trace(
        n_frames=6,
        exposure_time_s=0.05,
        seed=55,
        residual_x_sigma_as=0.01,
        residual_y_sigma_as=0.01,
        residual_pa_sigma_deg=1e-4,
    )

    assert first == second
    assert first_manifest["residual_jitter_sigmas"] == second_manifest["residual_jitter_sigmas"]


def test_expand_linear_drift_frame_values():
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "examples" / "recipes" / "observation_subblock_inference.py"
    spec = importlib.util.spec_from_file_location("observation_subblock_inference_for_drift_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    expanded = module.expand_linear_drift_frame_values(
        frame_times_s=[0.05, 0.15, 0.25],
        compact=[1.0, 2.0, -1.0, 4.0, 90.0, 0.5],
    )
    expected_centered = np.asarray([-0.1, 0.0, 0.1])
    np.testing.assert_allclose(expanded[:, 0], 1.0 + 2.0 * expected_centered)
    np.testing.assert_allclose(expanded[:, 1], -1.0 + 4.0 * expected_centered)
    np.testing.assert_allclose(expanded[:, 2], 90.0 + 0.5 * expected_centered)


def _load_inference_module(name: str = "observation_subblock_inference_temporal_test"):
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "examples" / "recipes" / "observation_subblock_inference.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _registration_layout(module, *, kind: str = "linear_drift_residual_jitter_prior", n_frame: int = 5):
    specs = tuple(
        module.ActiveKeySpec(canonical=key, address=None, kind="primitive")
        for key in module.LINEAR_DRIFT_REGISTRATION_KEYS
    )
    return module.ActiveStateLayout(
        frame_specs=specs,
        shared_specs=(),
        n_frame=n_frame,
        temporal_kind=kind,
        frame_times_s=tuple((np.arange(n_frame) + 0.5) * 0.1),
    )


def test_linear_fit_residuals_zero_for_exact_line():
    module = _load_inference_module("observation_subblock_inference_residual_zero")
    times = module.jnp.asarray([0.05, 0.15, 0.25, 0.35])
    centered = times - module.jnp.mean(times)
    values = module.jnp.stack(
        (1.0 + 2.0 * centered, -1.0 + 3.0 * centered, 90.0 - 0.1 * centered),
        axis=1,
    )
    residuals = np.asarray(module._linear_fit_residuals(values, times))
    np.testing.assert_allclose(residuals, 0.0, atol=1e-6)


def test_residual_prior_penalty_positive_and_sigma_scales_quadratically():
    module = _load_inference_module("observation_subblock_inference_residual_penalty")
    layout = _registration_layout(module, n_frame=5)
    values = np.zeros((5, 3), dtype=float)
    values[:, 0] = [0.0, 0.01, -0.01, 0.01, 0.0]
    state = module.ActiveState(frame=module.jnp.asarray(values), shared=module.jnp.asarray([]))
    objective_cfg = {"subblock_reduce": "sum"}
    cfg_small = {
        "kind": "linear_drift_residual_jitter_prior",
        "residual_prior": {
            "source.x_position_as": 0.01,
            "source.y_position_as": 0.01,
            "source.position_angle_deg": 1e-4,
        },
        "reduce": "sum",
    }
    cfg_large = {
        **cfg_small,
        "residual_prior": {
            "source.x_position_as": 0.02,
            "source.y_position_as": 0.01,
            "source.position_angle_deg": 1e-4,
        },
    }
    penalty_small = float(
        module._build_temporal_term_fn(
            {"frame_model": cfg_small},
            layout=layout,
            objective_cfg=objective_cfg,
        )(state)
    )
    penalty_large = float(
        module._build_temporal_term_fn(
            {"frame_model": cfg_large},
            layout=layout,
            objective_cfg=objective_cfg,
        )(state)
    )
    assert penalty_small > 0.0
    assert penalty_large == pytest.approx(penalty_small / 4.0)


def test_residual_prior_validation_errors_and_theta_size():
    module = _load_inference_module("observation_subblock_inference_residual_validation")
    layout = _registration_layout(module, n_frame=4)
    assert layout.theta_size == 12
    bad_cfg = {
        "kind": "linear_drift_residual_jitter_prior",
        "residual_prior": {
            "source.x_position_as": 0.01,
            "source.y_position_as": 0.01,
        },
    }
    with pytest.raises(ValueError, match="source.position_angle_deg"):
        module._build_temporal_term_fn(
            {"frame_model": bad_cfg},
            layout=layout,
            objective_cfg={"subblock_reduce": "mean"},
        )
    bad_layout = module.ActiveStateLayout(
        frame_specs=(
            module.ActiveKeySpec(canonical="source.x_position_as", address=None, kind="primitive"),
        ),
        shared_specs=(),
        n_frame=4,
        temporal_kind="linear_drift_residual_jitter_prior",
        frame_times_s=(0.0, 1.0, 2.0, 3.0),
    )
    with pytest.raises(ValueError, match="canonical registration"):
        module._build_temporal_term_fn(
            {"frame_model": {
                "kind": "linear_drift_residual_jitter_prior",
                "residual_prior": {"source.x_position_as": 0.01},
            }},
            layout=bad_layout,
            objective_cfg={"subblock_reduce": "mean"},
        )


def test_default_plan_and_dry_run_outputs(tmp_path: Path):
    module = _load_module()
    result = module.main(
        [
            "--results-root",
            str(tmp_path),
            "--run-name",
            "dry",
            "--n-frames",
            "3",
            "--case-filter",
            "drift_truth__linear_drift_fit",
            "--dry-run",
        ]
    )
    run_root = Path(result["run_root"])

    assert (run_root / "manifest.json").exists()
    assert (run_root / "comparison_plan.csv").exists()
    assert (run_root / "comparison_plan.json").exists()
    assert (
        run_root
        / "cases"
        / "drift_truth__linear_drift_fit"
        / "trace"
        / "trace_truth_manifest.json"
    ).exists()


def test_full_plan_includes_residual_prior_and_config(tmp_path: Path):
    module = _load_module()
    cases = module.default_case_specs(
        seed=42,
        noise_mode="disabled",
        init_mode="truth",
        full=True,
    )
    names = {case.case_name for case in cases}
    assert "iid50_truth__independent_fit" in names
    assert "drift_resid10mas_truth__residual_prior_fit" in names
    case = next(case for case in cases if case.case_name == "drift_resid10mas_truth__residual_prior_fit")
    trace = tmp_path / "trace.csv"
    trace.write_text(
        "frame_index,time_s,source.x_position_as,source.y_position_as,source.position_angle_deg\n",
        encoding="utf-8",
    )
    _, args = _args_for_optimizer()
    optimizer = module._optimizer_settings_for_case(
        case,
        n_frames=3,
        noise_mode="disabled",
        args=args,
    )
    _, inference_path = module._case_configs(
        case=case,
        case_root=tmp_path,
        trace_csv=trace,
        n_frames=3,
        exposure_time_s=0.05,
        system_preset="SHERA_TESTBED_3P",
        optimizer=optimizer,
    )
    cfg = json.loads(inference_path.read_text(encoding="utf-8"))
    frame_model = cfg["experiment"]["inference"]["temporal"]["frame_model"]
    assert frame_model["kind"] == "linear_drift_residual_jitter_prior"
    assert frame_model["residual_prior"]["source.x_position_as"]["sigma"] == pytest.approx(0.01)
    assert frame_model["reduce"] == "match_subblock_reduce"
    assert cfg["experiment"]["inference"]["optimizer"]["n_iter"] == 200
    assert cfg["experiment"]["inference"]["optimizer"]["base_lr"] == pytest.approx(0.9)
    schedule = cfg["experiment"]["inference"]["optimizer"]["schedule"]
    assert schedule["kind"] == "linear_warmup"
    assert schedule["warmup_steps"] == 10
    assert schedule["start_factor"] == pytest.approx(0.125)
    preconditioning = cfg["experiment"]["inference"]["optimizer"]["preconditioning"]
    assert preconditioning["method"] == "structured_residual_prior_diag"
    assert preconditioning["max_dense_dim"] == 20
    assert preconditioning["allow_dense_image_hessian"] is False
    assert cfg["experiment"]["inference"]["init"]["frame"]["mode"] == "from_truth_trace"


def test_truth_plus_offset_init_config_written(tmp_path: Path):
    module = _load_module()
    case = module.CaseSpec(
        case_name="offset",
        truth_model="linear_drift",
        fit_model="independent",
        init_mode="truth_plus_offset",
    )
    trace = tmp_path / "trace.csv"
    trace.write_text(
        "frame_index,time_s,source.x_position_as,source.y_position_as,source.position_angle_deg\n",
        encoding="utf-8",
    )
    _, args = _args_for_optimizer()
    optimizer = module._optimizer_settings_for_case(
        case,
        n_frames=3,
        noise_mode="disabled",
        args=args,
    )
    _, inference_path = module._case_configs(
        case=case,
        case_root=tmp_path,
        trace_csv=trace,
        n_frames=3,
        exposure_time_s=0.05,
        system_preset="SHERA_TESTBED_3P",
        optimizer=optimizer,
    )
    cfg = json.loads(inference_path.read_text(encoding="utf-8"))
    frame_init = cfg["experiment"]["inference"]["init"]["frame"]
    assert frame_init["mode"] == "from_truth_trace"
    assert frame_init["offsets"]["source.x_position_as"] == pytest.approx(1.0e-3)
    assert frame_init["offsets"]["source.y_position_as"] == pytest.approx(1.0e-3)
    assert frame_init["offsets"]["source.position_angle_deg"] == pytest.approx(1.0e-5)


def test_noise_enabled_prefers_render_variance(tmp_path: Path):
    module = _load_module()
    case = module.CaseSpec(
        case_name="noise",
        truth_model="linear_drift",
        fit_model="independent",
        noise_mode="enabled",
    )
    trace = tmp_path / "trace.csv"
    trace.write_text(
        "frame_index,time_s,source.x_position_as,source.y_position_as,source.position_angle_deg\n",
        encoding="utf-8",
    )
    _, args = _args_for_optimizer()
    optimizer = module._optimizer_settings_for_case(
        case,
        n_frames=3,
        noise_mode="enabled",
        args=args,
    )
    _, inference_path = module._case_configs(
        case=case,
        case_root=tmp_path,
        trace_csv=trace,
        n_frames=3,
        exposure_time_s=0.05,
        system_preset="SHERA_TESTBED_3P",
        optimizer=optimizer,
    )
    render_dir = tmp_path / "render"
    render_dir.mkdir()
    variance = render_dir / "variance.fits"
    variance.write_bytes(b"placeholder")
    (render_dir / "manifest.json").write_text(
        json.dumps({"artifacts": {"variance_fits": variance.name}}),
        encoding="utf-8",
    )
    status = module._patch_inference_variance_config(
        inference_path,
        case=case,
        case_root=tmp_path,
    )
    cfg = json.loads(inference_path.read_text(encoding="utf-8"))
    assert status["variance_model_used"] == "provided_cube"
    assert cfg["experiment"]["inference"]["objective"]["noise_model"]["variance_model"] == "provided_cube"


def test_dry_run_rows_include_schur_routing_fields(tmp_path: Path):
    module = _load_module()
    result = module.main(
        [
            "--results-root",
            str(tmp_path),
            "--run-name",
            "routing_dry",
            "--n-frames",
            "20",
            "--full-default-matrix",
            "--dry-run",
        ]
    )
    run_root = Path(result["run_root"])
    rows = list(csv.DictReader((run_root / "aggregate" / "case_metrics.csv").open("r", encoding="utf-8", newline="")))
    assert rows
    by_name = {row["case_name"]: row for row in rows}
    assert by_name["iid50_truth__independent_fit"]["schur_curvature_method_requested"] == "auto"
    assert by_name["iid50_truth__independent_fit"]["schur_max_dense_dim_effective"] == "40"
    assert by_name["iid50_truth__residual_prior_fit"]["schur_curvature_method_requested"] == "structured_residual_prior"
    assert int(by_name["iid50_truth__residual_prior_fit"]["schur_max_dense_dim_effective"]) == 40


def test_run_case_passes_requested_n_frames_to_schur(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = _load_module()
    case = module.CaseSpec(
        case_name="dim_case",
        truth_model="linear_drift_residual_jitter",
        fit_model="linear_drift_residual_jitter_prior",
        noise_mode="disabled",
        init_mode="truth",
    )

    calls: dict[str, object] = {}

    def _fake_generate_obs_subblock(**kwargs):
        render_dir = tmp_path / "run" / "cases" / case.case_name / "render"
        render_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = render_dir / "manifest.json"
        manifest_path.write_text(json.dumps({"artifacts": {}}), encoding="utf-8")
        cube = render_dir / "obs_subblock_fake_cube.fits"
        cube.write_bytes(b"cube")
        trace = render_dir / "obs_subblock_fake_frame_truth.csv"
        trace.write_text("frame_index,time_s\n", encoding="utf-8")
        return {"artifacts": {"cube_fits": str(cube), "frame_truth_csv": str(trace)}}

    def _fake_inference_main(argv):
        _ = argv
        inf_dir = tmp_path / "run" / "cases" / case.case_name / "inference"
        inf_dir.mkdir(parents=True, exist_ok=True)
        recovered = inf_dir / "obs_subblock_inference_fake_recovered_trace.csv"
        recovered.write_text(
            "frame_index,time_s,source.x_position_as,source.y_position_as,source.position_angle_deg\n"
            "0,0.05,0,0,90\n1,0.15,0,0,90\n2,0.25,0,0,90\n",
            encoding="utf-8",
        )
        manifest = inf_dir / "manifest.json"
        manifest.write_text(json.dumps({"metrics": {"chi2": {"initial_model": {}, "final_model": {}}}}), encoding="utf-8")
        return {
            "artifacts": {"recovered_trace_csv": str(recovered), "manifest_json": str(manifest)},
            "final_loss": 1.0,
            "chi2": {"initial_model": {}, "final_model": {}},
        }

    def _fake_schur(**kwargs):
        calls["n_frames"] = kwargs.get("n_frames")
        study_dir = tmp_path / "run" / "cases" / case.case_name / "study" / "schur_summary"
        study_dir.mkdir(parents=True, exist_ok=True)
        summary_json = study_dir / "subblock_summary.json"
        summary_json.write_text("{}", encoding="utf-8")
        (study_dir / "subblock_summary_matrices.npz").write_bytes(b"npz")
        return {"schur_summary": {"summary_json_path": str(summary_json)}}

    monkeypatch.setattr(module.observation_subblock, "generate_obs_subblock", _fake_generate_obs_subblock)
    monkeypatch.setattr(module.observation_subblock_inference, "main", _fake_inference_main)
    monkeypatch.setattr(module.run_obs_subblock_study, "run_obs_subblock_study", _fake_schur)
    monkeypatch.setattr(module, "_schur_metrics", lambda summary_json, theta_keys: {})
    monkeypatch.setattr(module, "_patch_inference_variance_config", lambda inference_path, case, case_root: {})

    _, args = _args_for_optimizer()
    row = module._run_case(
        case=case,
        run_root=tmp_path / "run",
        n_frames=3,
        exposure_time_s=0.05,
        system_preset="SHERA_TESTBED_3P",
        dry_run=False,
        resume=False,
        theta_keys=module.THETA_KEYS_DEFAULT,
        reference_diagnostics_profile="basic",
        max_dense_dim=40,
        args=args,
    )
    assert row["status"] == "completed"
    assert calls["n_frames"] == 3


def test_cli_optimizer_overrides_flow_into_generated_config(tmp_path: Path):
    module = _load_module()
    result = module.main(
        [
            "--results-root",
            str(tmp_path),
            "--run-name",
            "opt_override",
            "--n-frames",
            "3",
            "--noise",
            "enabled",
            "--full-default-matrix",
            "--case-filter",
            "drift_resid10mas_truth__residual_prior_fit",
            "--dry-run",
            "--optimizer-base-lr",
            "0.3",
            "--optimizer-n-iter",
            "77",
            "--schedule-warmup-steps",
            "3",
            "--schedule-start-factor",
            "0.5",
        ]
    )
    run_root = Path(result["run_root"])
    cfg = json.loads(
        (
            run_root
            / "cases"
            / "drift_resid10mas_truth__residual_prior_fit"
            / "inference_config.json"
        ).read_text(encoding="utf-8")
    )
    optimizer = cfg["experiment"]["inference"]["optimizer"]
    assert optimizer["base_lr"] == pytest.approx(0.3)
    assert optimizer["n_iter"] == 77
    assert optimizer["schedule"]["warmup_steps"] == 3
    assert optimizer["schedule"]["start_factor"] == pytest.approx(0.5)


def test_linear_drift_rejects_unsupported_compact_shape():
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "examples" / "recipes" / "observation_subblock_inference.py"
    spec = importlib.util.spec_from_file_location("observation_subblock_inference_shape_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with pytest.raises(ValueError, match="shape"):
        module.expand_linear_drift_frame_values(frame_times_s=[0.0], compact=[1.0, 2.0])
