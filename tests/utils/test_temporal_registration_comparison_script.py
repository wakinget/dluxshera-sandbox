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
    _, inference_path = module._case_configs(
        case=case,
        case_root=tmp_path,
        trace_csv=trace,
        n_frames=3,
        exposure_time_s=0.05,
        system_preset="SHERA_TESTBED_3P",
    )
    cfg = json.loads(inference_path.read_text(encoding="utf-8"))
    frame_model = cfg["experiment"]["inference"]["temporal"]["frame_model"]
    assert frame_model["kind"] == "linear_drift_residual_jitter_prior"
    assert frame_model["residual_prior"]["source.x_position_as"]["sigma"] == pytest.approx(0.01)
    assert frame_model["reduce"] == "match_subblock_reduce"


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
    _, inference_path = module._case_configs(
        case=case,
        case_root=tmp_path,
        trace_csv=trace,
        n_frames=3,
        exposure_time_s=0.05,
        system_preset="SHERA_TESTBED_3P",
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
