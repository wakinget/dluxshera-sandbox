from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "scripts"
    / "run_obs_subblock_monte_carlo.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("run_obs_subblock_monte_carlo_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _config(module, tmp_path: Path, **overrides):
    payload = {
        "run_name": "mc test",
        "results_root": tmp_path,
        "n_trials": 3,
        "max_workers": 1,
        "seed": 42,
        "seed_policy": "different_jitter_different_noise",
        "n_frames": 10,
        "noise": "enabled",
        "theta_keys": (
            "source.separation_as",
            "source.log_flux_total",
        ),
        "phi_ref": "truth_when_available",
        "schur_curvature_method": "auto",
        "variance_floor": 1.0,
        "plots": False,
    }
    payload.update(overrides)
    return module.MonteCarloRunConfig(**payload)


def _write_summary(
    path: Path,
    *,
    labels=("a", "b"),
    theta_ref=(0.0, 0.0),
    score=(0.0, 0.0),
    metadata: dict | None = None,
):
    info = np.asarray([[4.0, 1.0], [1.0, 9.0]], dtype=float)
    score_arr = np.asarray(score, dtype=float)
    theta_ref_arr = np.asarray(theta_ref, dtype=float)
    npz_path = path.with_name("subblock_summary_matrices.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        theta_ref=theta_ref_arr,
        phi_ref=np.zeros(0),
        reduced_information=info,
        reduced_score=score_arr,
    )
    payload = {
        "schema_version": "image_backed_subblock_summary.v1",
        "subblock_id": path.parent.parent.parent.name,
        "summary_kind": "image_backed_schur",
        "theta_labels": list(labels),
        "phi_labels": [],
        "combined_labels": list(labels),
        "theta_ref": theta_ref_arr.tolist(),
        "phi_ref": [],
        "dimensions": {"n_theta": len(labels), "n_phi": 0, "combined_dim": len(labels)},
        "summary_diagnostics": {"curvature_method_used": "synthetic"},
        "matrix_artifact_path": npz_path.name,
        "metadata": {"structured_curvature_used": False, **dict(metadata or {})},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_seed_policies_are_stable_and_policy_specific(tmp_path: Path):
    module = _load_module()
    cfg = _config(module, tmp_path)
    plan_a = module.build_trial_plan(cfg)
    plan_b = module.build_trial_plan(cfg)
    assert [(p.trace_seed, p.noise_seed) for p in plan_a] == [
        (p.trace_seed, p.noise_seed) for p in plan_b
    ]
    assert len({p.trace_seed for p in plan_a}) == 3
    assert len({p.noise_seed for p in plan_a}) == 3

    same_trace = module.build_trial_plan(
        _config(module, tmp_path, seed_policy="same_jitter_different_noise")
    )
    assert len({p.trace_seed for p in same_trace}) == 1
    assert len({p.noise_seed for p in same_trace}) == 3

    same_noise = module.build_trial_plan(
        _config(module, tmp_path, seed_policy="different_jitter_same_noise")
    )
    assert len({p.trace_seed for p in same_noise}) == 3
    assert len({p.noise_seed for p in same_noise}) == 1


def test_mc_log_prefix_and_quiet_mode(capsys):
    module = _load_module()
    module.mc_log("plan.ready", run_name="abc", trials=2)
    out = capsys.readouterr().out
    assert out.startswith("[obs_subblock_mc] plan.ready")
    assert "run_name=abc" in out
    assert "trials=2" in out

    module.mc_log("heartbeat", quiet=True, completed="0/2")
    assert capsys.readouterr().out == ""

    module.mc_log("trial.failed", quiet=True, force=True, trial_id=1)
    assert "[obs_subblock_mc] trial.failed" in capsys.readouterr().out


def test_tail_text_file_missing_last_lines_and_truncation(tmp_path: Path):
    module = _load_module()
    assert module.tail_text_file(tmp_path / "missing.log", n_lines=2) == ()
    path = tmp_path / "trial.log"
    path.write_text("one\n" + "x" * 300 + "\nthree\n", encoding="utf-8")
    lines = module.tail_text_file(path, n_lines=2, max_chars=20)
    assert len(lines) == 2
    assert lines[0] == "x" * 17 + "..."
    assert lines[1] == "three"


def test_plan_generation_and_command_construction(tmp_path: Path):
    module = _load_module()
    cfg = _config(module, tmp_path, run_name="space safe")
    plan = module.build_trial_plan(cfg)
    assert len(plan) == 3
    assert cfg.run_name == "space safe"
    assert len({p.case_name for p in plan}) == 3
    assert all(str(p.expected_summary_json).startswith(str(cfg.run_root)) for p in plan)

    command = module.build_trial_command(plan[0], cfg)
    joined = " ".join(command)
    assert "examples/scripts/run_obs_subblock_study.py" in joined
    assert "--mode schur_summary" in joined
    assert "--n-frames 10" in joined
    assert "--noise enabled" in joined
    assert "--theta-keys source.separation_as,source.log_flux_total" in joined
    assert "--phi-ref truth_when_available" in joined
    assert "--schur-curvature-method auto" in joined
    assert "--max-dense-dim 40" in joined
    assert "--variance-floor 1.0" in joined
    assert "--trace-seed" in joined
    assert "--render-seed" in joined
    assert "--reference-diagnostics-profile none" in joined
    assert "--reference-optimizer-kind sgd" in joined
    assert "--reference-base-lr 0.7" in joined
    assert "--reference-n-iter 80" in joined
    assert "--schur-frame-quality-policy warn" in joined
    assert "--schur-frame-chi2-threshold 5.0" in joined

    module.write_command_file(plan[0], command)
    text = plan[0].command_path.read_text(encoding="utf-8")
    assert "PYTHONPATH=" in text
    assert "run_obs_subblock_study.py" in text
    assert "--max-dense-dim 40" in text
    assert "--reference-base-lr 0.7" in text


def test_frame_quality_cli_command_plan_and_manifest(tmp_path: Path):
    module = _load_module()
    parser = module._build_parser()
    cfg = module.build_config_from_args(
        parser.parse_args(
            [
                "--run-name",
                "fq",
                "--results-root",
                str(tmp_path),
                "--schur-frame-quality-policy",
                "mask",
                "--schur-frame-chi2-threshold",
                "4.5",
                "--schur-frame-quality-missing",
                "error",
                "--schur-frame-mask-denominator",
                "kept",
                "--schur-frame-mask-min-good-frames",
                "2",
            ]
        )
    )
    plan = module.build_trial_plan(cfg)
    command = " ".join(module.build_trial_command(plan[0], cfg))
    assert "--schur-frame-quality-policy mask" in command
    assert "--schur-frame-chi2-threshold 4.5" in command
    assert "--schur-frame-quality-missing error" in command
    assert "--schur-frame-mask-denominator kept" in command
    assert "--schur-frame-mask-min-good-frames 2" in command

    module.write_run_plan_csv(cfg.run_root / "run_plan.csv", plan)
    row = _read_csv(cfg.run_root / "run_plan.csv")[0]
    assert row["schur_frame_quality_policy"] == "mask"
    assert row["schur_frame_chi2_threshold"] == "4.5"
    assert row["schur_frame_quality_missing"] == "error"
    assert row["schur_frame_mask_denominator"] == "kept"

    module.write_manifest(cfg, plan)
    manifest = json.loads((cfg.run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["study_defaults"]["schur_frame_quality_policy"] == "mask"
    assert manifest["study_defaults"]["schur_frame_chi2_threshold"] == 4.5


def test_memory_diagnostics_cli_is_forwarded_to_trial_command(tmp_path: Path):
    module = _load_module()
    cfg = _config(module, tmp_path, memory_diagnostics=True)
    plan = module.build_trial_plan(cfg)
    command = " ".join(module.build_trial_command(plan[0], cfg))
    assert "--memory-diagnostics" in command

    parser = module._build_parser()
    parsed = parser.parse_args(
        [
            "--run-name",
            "memory",
            "--results-root",
            str(tmp_path),
            "--memory-diagnostics",
            "--memory-progress-tail-lines",
            "5",
        ]
    )
    built = module.build_config_from_args(parsed)
    assert built.memory_diagnostics is True
    assert built.memory_progress_tail_lines == 5


def test_return_code_negative_nine_is_probable_sigkill():
    module = _load_module()
    failure_class, failure_hint = module.classify_subprocess_failure(-9)
    assert failure_class == "probable_sigkill"
    assert failure_hint == "possible_memory_pressure_or_external_kill"


def test_reference_diagnostics_profile_cli_config_and_manifest(tmp_path: Path):
    module = _load_module()
    cfg = _config(module, tmp_path, reference_diagnostics_profile="basic")
    plan = module.build_trial_plan(cfg)
    command = module.build_trial_command(plan[0], cfg)
    assert "--reference-diagnostics-profile basic" in " ".join(command)
    assert plan[0].reference_diagnostics_profile == "basic"

    module.write_run_plan_csv(cfg.run_root / "run_plan.csv", plan)
    plan_rows = _read_csv(cfg.run_root / "run_plan.csv")
    assert plan_rows[0]["reference_diagnostics_profile"] == "basic"

    module.write_manifest(cfg, plan)
    manifest = json.loads((cfg.run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["study_defaults"]["reference_diagnostics_profile"] == "basic"


def test_reference_frame_quality_extraction_resolves_relative_manifest(tmp_path: Path):
    module = _load_module()
    summary_path = tmp_path / "case" / "study" / "schur_summary" / "subblock_summary.json"
    manifest_path = summary_path.parent / "reference_inference" / "inference" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "chi2": {
                    "final_model": {
                        "per_frame_reduced_chi2": [1.0, 6.5, 2.0],
                        "block_reduced_chi2": 1.7,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "metadata": {
            "recovered_reference": {
                "manifest_json": "reference_inference/inference/manifest.json"
            }
        }
    }
    result = module.extract_reference_frame_quality_from_summary_payload(
        payload,
        summary_json_path=summary_path,
        threshold=5.0,
    )
    assert result["reference_frame_quality_source"] == "found"
    assert result["reference_final_max_frame_reduced_chi2"] == 6.5
    assert result["reference_final_median_frame_reduced_chi2"] == 2.0
    assert result["reference_final_block_reduced_chi2"] == 1.7
    assert result["reference_failed_frame_count_chi2_gt_threshold"] == 1
    assert result["reference_failed_frame_indices_chi2_gt_threshold"] == "1"
    assert result["fit_warning"] == "reference_final_max_frame_reduced_chi2_high"


def test_reference_frame_quality_extraction_missing_manifest_is_nonfatal(tmp_path: Path):
    module = _load_module()
    result = module.extract_reference_frame_quality_from_summary_payload(
        {"metadata": {"recovered_reference": {"manifest_json": "missing.json"}}},
        summary_json_path=tmp_path / "subblock_summary.json",
        threshold=5.0,
    )
    assert result["reference_frame_quality_source"] == "missing_manifest"
    assert result["reference_final_max_frame_reduced_chi2"] == ""
    assert result["fit_warning"] == ""


def test_reference_optimizer_overrides_appear_in_command_plan_and_manifest(tmp_path: Path):
    module = _load_module()
    cfg = _config(
        module,
        tmp_path,
        reference_optimizer_kind="adam",
        reference_base_lr=1.0e-3,
        reference_n_iter=300,
        reference_optimizer_kwargs={"b1": 0.8, "b2": 0.999, "eps": 1.0e-8},
        reference_preconditioning_enabled=True,
        reference_preconditioning_method="auto",
        reference_preconditioning_reference="initial",
        reference_preconditioning_damping=1.0e-6,
        reference_preconditioning_eig_floor_rel=1.0e-6,
        reference_preconditioning_eig_floor_abs=1.0e-8,
        reference_preconditioning_lr_clip=(0.1, 10.0),
    )
    plan = module.build_trial_plan(cfg)
    command = " ".join(module.build_trial_command(plan[0], cfg))
    assert "--reference-optimizer-kind adam" in command
    assert "--reference-base-lr 0.001" in command
    assert "--reference-n-iter 300" in command
    assert "--reference-optimizer-kwarg b1=0.8" in command
    assert "--reference-optimizer-kwarg b2=0.999" in command
    assert "--reference-optimizer-kwarg eps=1e-08" in command
    assert "--reference-preconditioning-enabled" in command
    assert "--reference-preconditioning-method auto" in command
    assert "--reference-preconditioning-reference initial" in command
    assert "--reference-preconditioning-damping 1e-06" in command
    assert "--reference-preconditioning-eig-floor-rel 1e-06" in command
    assert "--reference-preconditioning-eig-floor-abs 1e-08" in command
    assert "--reference-preconditioning-lr-clip 0.1,10" in command

    module.write_run_plan_csv(cfg.run_root / "run_plan.csv", plan)
    row = _read_csv(cfg.run_root / "run_plan.csv")[0]
    assert row["max_dense_dim"] == "40"
    assert row["reference_optimizer_kind"] == "adam"
    assert row["reference_base_lr"] == "0.001"
    assert row["reference_n_iter"] == "300"
    assert row["reference_preconditioning_enabled"] == "True"
    assert row["reference_preconditioning_method"] == "auto"
    assert row["reference_preconditioning_reference"] == "initial"
    assert row["reference_preconditioning_lr_clip"] == "0.1,10"

    module.write_manifest(cfg, plan)
    manifest = json.loads((cfg.run_root / "manifest.json").read_text(encoding="utf-8"))
    payload = manifest["reference_optimizer_overrides"]
    assert payload["kind"] == "adam"
    assert payload["base_lr"] == pytest.approx(1.0e-3)
    assert payload["n_iter"] == 300
    assert payload["kwargs"] == {"b1": 0.8, "b2": 0.999, "eps": 1.0e-8}
    assert payload["preconditioning"]["enabled"] is True
    assert payload["preconditioning"]["lr_clip"] == [0.1, 10.0]


def test_reference_schedule_override_appears_in_command_plan_and_manifest(tmp_path: Path):
    module = _load_module()
    cfg = _config(
        module,
        tmp_path,
        reference_schedule={
            "kind": "linear_warmup",
            "warmup_steps": 8,
            "start_factor": 0.25,
        },
    )
    plan = module.build_trial_plan(cfg)
    command = " ".join(module.build_trial_command(plan[0], cfg))
    assert "--reference-schedule-kind linear_warmup" in command
    assert "--reference-schedule-warmup-steps 8" in command
    assert "--reference-schedule-start-factor 0.25" in command

    module.write_run_plan_csv(cfg.run_root / "run_plan.csv", plan)
    row = _read_csv(cfg.run_root / "run_plan.csv")[0]
    assert row["reference_schedule_kind"] == "linear_warmup"
    assert json.loads(row["reference_schedule_json"]) == {
        "kind": "linear_warmup",
        "start_factor": 0.25,
        "warmup_steps": 8,
    }

    module.write_manifest(cfg, plan)
    manifest = json.loads((cfg.run_root / "manifest.json").read_text(encoding="utf-8"))
    payload = manifest["reference_optimizer_overrides"]
    assert payload["schedule"] == {
        "kind": "linear_warmup",
        "warmup_steps": 8,
        "start_factor": 0.25,
    }


def test_reference_early_stopping_appears_in_trial_command(tmp_path: Path):
    module = _load_module()
    cfg = _config(
        module,
        tmp_path,
        reference_early_stopping_enabled=True,
        reference_early_stopping_min_iter=10,
        reference_early_stopping_patience=4,
        reference_early_stopping_loss_rtol=1.0e-8,
    )
    command = " ".join(module.build_trial_command(module.build_trial_plan(cfg)[0], cfg))

    assert "--reference-early-stopping" in command
    assert "--reference-early-stopping-min-iter 10" in command
    assert "--reference-early-stopping-patience 4" in command
    assert "--reference-early-stopping-loss-rtol 1e-08" in command


def test_default_mc_optimizer_and_dense_guard_are_recorded(tmp_path: Path):
    module = _load_module()
    cfg = _config(module, tmp_path, run_name="defaults")
    plan = module.build_trial_plan(cfg)
    command = " ".join(module.build_trial_command(plan[0], cfg))
    assert "--max-dense-dim 40" in command
    assert "--reference-optimizer-kind sgd" in command
    assert "--reference-base-lr 0.7" in command
    assert "--reference-n-iter 80" in command

    module.write_run_plan_csv(cfg.run_root / "run_plan.csv", plan)
    row = _read_csv(cfg.run_root / "run_plan.csv")[0]
    assert row["max_dense_dim"] == "40"
    assert row["reference_optimizer_kind"] == "sgd"
    assert row["reference_base_lr"] == "0.7"
    assert row["reference_n_iter"] == "80"
    assert row["reference_schedule_kind"] == ""
    assert row["reference_schedule_json"] == ""

    module.write_manifest(cfg, plan)
    manifest = json.loads((cfg.run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["study_defaults"]["max_dense_dim"] == 40
    assert manifest["reference_optimizer_overrides"]["kind"] == "sgd"
    assert manifest["reference_optimizer_overrides"]["base_lr"] == pytest.approx(0.7)
    assert manifest["reference_optimizer_overrides"]["n_iter"] == 80
    assert manifest["reference_optimizer_overrides"]["schedule"] is None


def test_reference_optimizer_config_file_and_cli_precedence(tmp_path: Path):
    module = _load_module()
    config_path = tmp_path / "optimizer_config.json"
    config_path.write_text(
        json.dumps(
            {
                "run": {"run_name": "optimizer_config", "results_root": str(tmp_path)},
                "trial": {
                    "max_dense_dim": 55,
                    "reference_optimizer": {
                        "kind": "adam",
                        "base_lr": 0.001,
                        "n_iter": 300,
                        "kwargs": {"b1": 0.8},
                        "schedule": {
                            "kind": "linear_warmup",
                            "warmup_steps": 8,
                            "start_factor": 0.25,
                        },
                        "preconditioning": {
                            "enabled": True,
                            "method": "auto",
                            "reference": "initial",
                            "lr_clip": [0.1, 10.0],
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    parser = module._build_parser()
    cfg = module.build_config_from_args(parser.parse_args(["--config", str(config_path)]))
    assert cfg.reference_optimizer_kind == "adam"
    assert cfg.reference_base_lr == pytest.approx(0.001)
    assert cfg.reference_n_iter == 300
    assert cfg.max_dense_dim == 55
    assert cfg.reference_optimizer_kwargs == {"b1": 0.8}
    assert cfg.reference_schedule == {
        "kind": "linear_warmup",
        "warmup_steps": 8,
        "start_factor": 0.25,
    }
    assert cfg.reference_preconditioning_enabled is True
    assert cfg.reference_preconditioning_reference == "initial"
    assert cfg.reference_preconditioning_lr_clip == (0.1, 10.0)

    override = module.build_config_from_args(
        parser.parse_args(
            [
                "--config",
                str(config_path),
                "--reference-optimizer-kind",
                "adam",
                "--reference-base-lr",
                "0.3",
                "--reference-n-iter",
                "100",
                "--reference-schedule-kind",
                "linear_warmup",
                "--reference-schedule-warmup-steps",
                "5",
                "--reference-schedule-start-factor",
                "0.2",
                "--max-dense-dim",
                "44",
                "--reference-preconditioning-reference",
                "truth_when_available",
            ]
        )
    )
    assert override.reference_optimizer_kind == "adam"
    assert override.reference_base_lr == pytest.approx(0.3)
    assert override.reference_n_iter == 100
    assert override.reference_schedule == {
        "kind": "linear_warmup",
        "warmup_steps": 5,
        "start_factor": 0.2,
    }
    assert override.max_dense_dim == 44
    assert override.reference_preconditioning_reference == "truth_when_available"


def test_reference_schedule_alias_config_shape_is_accepted(tmp_path: Path):
    module = _load_module()
    config_path = tmp_path / "schedule_alias_config.json"
    config_path.write_text(
        json.dumps(
            {
                "run": {"run_name": "schedule_alias", "results_root": str(tmp_path)},
                "trial": {
                    "reference_optimizer_schedule": {
                        "kind": "piecewise_constant",
                        "boundaries": [50, 150],
                        "factors": [1.0, 0.3, 0.1],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    parser = module._build_parser()
    cfg = module.build_config_from_args(parser.parse_args(["--config", str(config_path)]))
    assert cfg.reference_schedule == {
        "kind": "piecewise_constant",
        "boundaries": [50, 150],
        "factors": [1.0, 0.3, 0.1],
    }


def test_reference_diagnostics_profile_config_file_and_cli_precedence(tmp_path: Path):
    module = _load_module()
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "run": {"run_name": "config_profile", "results_root": str(tmp_path)},
                "trial": {"reference_diagnostics_profile": "review"},
            }
        ),
        encoding="utf-8",
    )
    parser = module._build_parser()
    cfg = module.build_config_from_args(parser.parse_args(["--config", str(config_path)]))
    assert cfg.reference_diagnostics_profile == "review"

    cfg_override = module.build_config_from_args(
        parser.parse_args(
            [
                "--config",
                str(config_path),
                "--reference-diagnostics-profile",
                "basic",
            ]
        )
    )
    assert cfg_override.reference_diagnostics_profile == "basic"


def test_reference_diagnostics_profile_invalid_value_fails_clearly(tmp_path: Path):
    module = _load_module()
    config_path = tmp_path / "bad_config.json"
    config_path.write_text(
        json.dumps(
            {
                "run": {"run_name": "bad_profile", "results_root": str(tmp_path)},
                "trial": {"reference_diagnostics_profile": "expensive"},
            }
        ),
        encoding="utf-8",
    )
    parser = module._build_parser()
    try:
        module.build_config_from_args(parser.parse_args(["--config", str(config_path)]))
    except ValueError as exc:
        assert "Unsupported --reference-diagnostics-profile" in str(exc)
        assert "none, basic, review, full" in str(exc)
    else:
        raise AssertionError("Expected invalid reference diagnostics profile to fail.")


def test_dry_run_writes_manifest_plan_status_and_commands(tmp_path: Path):
    module = _load_module()
    result = module.main(
        [
            "--run-name",
            "dry run",
            "--results-root",
            str(tmp_path),
            "--n-trials",
            "2",
            "--theta-keys",
            "source.separation_as,source.log_flux_total",
            "--phi-ref",
            "truth_when_available",
            "--dry-run",
            "--no-plots",
        ]
    )
    run_root = Path(result["run_root"])
    assert (run_root / "manifest.json").exists()
    assert len(_read_csv(run_root / "run_plan.csv")) == 2
    assert len(_read_csv(run_root / "run_status.csv")) == 2
    assert (run_root / "commands" / "trial_000000.sh").exists()
    assert not (run_root / "logs" / "trial_000000_stdout.log").exists()


def test_resume_skips_completed_trials(tmp_path: Path):
    module = _load_module()
    cfg = _config(module, tmp_path, n_trials=1, resume=True)
    plan = module.build_trial_plan(cfg)
    _write_summary(plan[0].expected_summary_json)
    result = module.run_trial_subprocess(plan[0], module.build_trial_command(plan[0], cfg), resume=True)
    assert result.status == "skipped_completed"
    assert result.return_code == 0


def test_run_trial_pool_emits_progress_and_heartbeat(monkeypatch, tmp_path: Path, capsys):
    module = _load_module()
    cfg = _config(
        module,
        tmp_path,
        n_trials=2,
        max_workers=2,
        progress_interval_s=0.01,
        tail_lines=1,
    )
    plan = module.build_trial_plan(cfg)
    commands = {spec.trial_id: module.build_trial_command(spec, cfg) for spec in plan}

    def fake_run_trial_subprocess(spec, command, *, resume=False):
        spec.stdout_log.parent.mkdir(parents=True, exist_ok=True)
        spec.stdout_log.write_text(f"trial {spec.trial_id} working\n", encoding="utf-8")
        time.sleep(0.05 if spec.trial_id == 0 else 0.03)
        status = "completed" if spec.trial_id == 0 else "failed"
        return module.MonteCarloTrialResult(
            trial_id=spec.trial_id,
            status=status,
            return_code=0 if status == "completed" else 1,
            started_at="start",
            finished_at="finish",
            elapsed_seconds=1.0,
            summary_json_path=spec.expected_summary_json if status == "completed" else None,
            matrix_npz_path=None,
            failure_reason=None if status == "completed" else "synthetic_failure",
        )

    monkeypatch.setattr(module, "run_trial_subprocess", fake_run_trial_subprocess)
    results = module.run_trial_pool(plan, cfg, commands=commands)
    out = capsys.readouterr().out
    assert "execution.start" in out
    assert "trial.start" in out
    assert "heartbeat" in out
    assert "trial.done" in out
    assert "trial.failed" in out
    assert "execution.done" in out
    assert "stream=stdout" in out
    assert results[0].status == "completed"
    assert results[1].status == "failed"
    assert (cfg.run_root / "run_status.csv").exists()
    assert (cfg.run_root / "progress.json").exists()


def test_run_trial_pool_reports_resume_skips(tmp_path: Path, capsys):
    module = _load_module()
    cfg = _config(module, tmp_path, n_trials=1, resume=True)
    plan = module.build_trial_plan(cfg)
    _write_summary(plan[0].expected_summary_json)
    commands = {plan[0].trial_id: module.build_trial_command(plan[0], cfg)}
    results = module.run_trial_pool(plan, cfg, commands=commands)
    out = capsys.readouterr().out
    assert "execution.start" in out
    assert "skipped=1" in out
    assert "execution.resume" in out
    assert "execution.done" in out
    assert results[0].status == "skipped_completed"


def test_run_trial_pool_quiet_suppresses_progress(monkeypatch, tmp_path: Path, capsys):
    module = _load_module()
    cfg = _config(module, tmp_path, n_trials=1, quiet=True, progress_interval_s=0.01)
    plan = module.build_trial_plan(cfg)
    commands = {plan[0].trial_id: module.build_trial_command(plan[0], cfg)}

    def fake_run_trial_subprocess(spec, command, *, resume=False):
        return module.MonteCarloTrialResult(
            trial_id=spec.trial_id,
            status="completed",
            return_code=0,
            started_at="start",
            finished_at="finish",
            elapsed_seconds=0.0,
            summary_json_path=spec.expected_summary_json,
            matrix_npz_path=None,
        )

    monkeypatch.setattr(module, "run_trial_subprocess", fake_run_trial_subprocess)
    results = module.run_trial_pool(plan, cfg, commands=commands)
    assert capsys.readouterr().out == ""
    assert results[0].status == "completed"


def test_aggregate_only_uses_existing_plan_and_records_failures(tmp_path: Path):
    module = _load_module()
    cfg = _config(module, tmp_path, run_name="aggregate_only_case", n_trials=2)
    plan = module.build_trial_plan(cfg)
    module.write_run_plan_csv(cfg.run_root / "run_plan.csv", plan)
    module.write_run_status_csv(
        cfg.run_root / "run_status.csv",
        plan,
        {
            0: module.MonteCarloTrialResult(
                trial_id=0,
                status="failed",
                return_code=2,
                started_at=None,
                finished_at=None,
                elapsed_seconds=None,
                summary_json_path=None,
                matrix_npz_path=None,
                failure_reason="synthetic_failure",
            )
        },
    )
    result = module.main(
        [
            "--run-name",
            "aggregate_only_case",
            "--results-root",
            str(tmp_path),
            "--aggregate-only",
            "--no-plots",
        ]
    )
    assert result["aggregate_summary"]["n_trials_planned"] == 2
    assert result["aggregate_summary"]["n_trials_failed"] == 1
    assert result["aggregate_summary"]["n_planned_not_run"] == 1
    failures = _read_csv(cfg.run_root / "aggregate" / "failed_trials.csv")
    assert any(row["failure_reason"] == "synthetic_failure" for row in failures)
    assert any(row["status"] == "planned_not_run" for row in failures)


def test_memory_failure_summary_from_synthetic_sigkill_status(tmp_path: Path):
    module = _load_module()
    cfg = _config(module, tmp_path, run_name="memory_fail", n_trials=1, plots=False)
    plan = module.build_trial_plan(cfg)
    spec = plan[0]
    spec.stdout_log.parent.mkdir(parents=True, exist_ok=True)
    spec.stdout_log.write_text("last stdout\n", encoding="utf-8")
    spec.stderr_log.write_text("last stderr\n", encoding="utf-8")
    spec.memory_diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    spec.memory_diagnostics_path.write_text(
        json.dumps({"stage": "structured_schur_reduce.start", "rss_mb": 123.0, "peak_rss_mb": 456.0})
        + "\n",
        encoding="utf-8",
    )
    result = module.MonteCarloTrialResult(
        trial_id=0,
        status="failed",
        return_code=-9,
        started_at="start",
        finished_at="finish",
        elapsed_seconds=12.0,
        summary_json_path=None,
        matrix_npz_path=None,
        failure_reason="subprocess_return_code_-9",
        failure_class="probable_sigkill",
        failure_hint="possible_memory_pressure_or_external_kill",
        memory_diagnostics_path=spec.memory_diagnostics_path,
    )
    summary = module.aggregate_schur_summary_trials(
        config=cfg,
        plan=plan,
        results={0: result},
    )
    assert summary["n_failed_probable_sigkill"] == 1
    rows = _read_csv(cfg.run_root / "aggregate" / "memory_failure_summary.csv")
    assert rows[0]["failure_class"] == "probable_sigkill"
    assert rows[0]["last_stdout_line"] == "last stdout"
    assert rows[0]["last_stderr_line"] == "last stderr"
    assert rows[0]["last_memory_stage"] == "structured_schur_reduce.start"
    assert rows[0]["subblock_summary_json_exists"] == "False"


def test_missing_memory_jsonl_does_not_crash_aggregation(tmp_path: Path):
    module = _load_module()
    cfg = _config(module, tmp_path, run_name="missing_memory", n_trials=1, plots=False)
    plan = module.build_trial_plan(cfg)
    result = module.MonteCarloTrialResult(
        trial_id=0,
        status="failed",
        return_code=-9,
        started_at=None,
        finished_at=None,
        elapsed_seconds=None,
        summary_json_path=None,
        matrix_npz_path=None,
        failure_reason="subprocess_return_code_-9",
        failure_class="probable_sigkill",
    )
    module.aggregate_schur_summary_trials(config=cfg, plan=plan, results={0: result})
    rows = _read_csv(cfg.run_root / "aggregate" / "memory_failure_summary.csv")
    assert rows[0]["last_memory_stage"] == ""


def test_fit_warning_reads_recovered_manifest_when_available(tmp_path: Path):
    module = _load_module()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "chi2": {
                    "final_model": {
                        "per_frame_reduced_chi2": [1.0, 6.5],
                        "block_reduced_chi2": 2.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    fields = module._fit_warning_from_recovered_manifest(
        {"metadata": {"recovered_reference": {"manifest_json": str(manifest_path)}}}
    )
    assert fields["reference_final_max_frame_reduced_chi2"] == pytest.approx(6.5)
    assert fields["fit_warning"] == "reference_final_max_frame_reduced_chi2_high"


def test_synthetic_summary_aggregation_writes_tables_and_whitened_residuals(tmp_path: Path):
    module = _load_module()
    cfg = _config(module, tmp_path, run_name="aggregate", n_trials=2, plots=False)
    plan = module.build_trial_plan(cfg)
    results = {}
    for index, spec in enumerate(plan):
        _write_summary(
            spec.expected_summary_json,
            labels=("a", "b"),
            theta_ref=(0.0, 0.0),
            score=(float(index), -float(index)),
        )
        results[spec.trial_id] = module.MonteCarloTrialResult(
            trial_id=spec.trial_id,
            status="completed",
            return_code=0,
            started_at=None,
            finished_at=None,
            elapsed_seconds=1.0,
            summary_json_path=spec.expected_summary_json,
            matrix_npz_path=spec.expected_matrix_npz,
        )
    summary = module.aggregate_schur_summary_trials(config=cfg, plan=plan, results=results)
    aggregate_root = cfg.run_root / "aggregate"
    assert summary["n_summaries_accepted"] == 2
    assert len(_read_csv(aggregate_root / "accepted_summary_paths.csv")) == 2
    assert len(_read_csv(aggregate_root / "summary_metrics.csv")) == 2
    assert len(_read_csv(aggregate_root / "matrix_diagonal_entries.csv")) == 4
    assert len(_read_csv(aggregate_root / "score_entries.csv")) == 4
    assert len(_read_csv(aggregate_root / "eigenvalue_metrics.csv")) == 4
    assert len(_read_csv(aggregate_root / "matrix_correlation_entries.csv")) == 8
    assert len(_read_csv(aggregate_root / "whitened_score_residuals.csv")) == 4
    corr = _read_csv(aggregate_root / "matrix_correlation_entries.csv")
    off_diag = [row for row in corr if row["i"] == "0" and row["j"] == "1"][0]
    assert np.isclose(float(off_diag["correlation"]), 1.0 / 6.0)


def test_missing_summary_is_rejected_with_clear_reason(tmp_path: Path):
    module = _load_module()
    cfg = _config(module, tmp_path, run_name="reject", n_trials=1, plots=False)
    plan = module.build_trial_plan(cfg)
    results = {
        0: module.MonteCarloTrialResult(
            trial_id=0,
            status="completed",
            return_code=0,
            started_at=None,
            finished_at=None,
            elapsed_seconds=1.0,
            summary_json_path=plan[0].expected_summary_json,
            matrix_npz_path=None,
        )
    }
    module.aggregate_schur_summary_trials(config=cfg, plan=plan, results=results)
    failures = _read_csv(cfg.run_root / "aggregate" / "failed_trials.csv")
    assert failures[0]["failure_reason"] == "missing_summary_json"


def test_plot_smoke_with_synthetic_summaries(tmp_path: Path):
    module = _load_module()
    cfg = _config(module, tmp_path, run_name="plots", n_trials=1, plots=True)
    plan = module.build_trial_plan(cfg)
    _write_summary(plan[0].expected_summary_json)
    results = {
        0: module.MonteCarloTrialResult(
            trial_id=0,
            status="completed",
            return_code=0,
            started_at=None,
            finished_at=None,
            elapsed_seconds=1.0,
            summary_json_path=plan[0].expected_summary_json,
            matrix_npz_path=plan[0].expected_matrix_npz,
        )
    }
    module.aggregate_schur_summary_trials(config=cfg, plan=plan, results=results)
    plots = cfg.run_root / "aggregate" / "plots"
    assert (plots / "s_diagonal_histograms.png").exists()
    assert (plots / "score_entry_histograms.png").exists()
    assert (plots / "score_norm_histogram.png").exists()
    assert (plots / "eigenvalue_spectrum_quantiles.png").exists()
    assert (plots / "correlation_mean_heatmap.png").exists()
    assert (plots / "correlation_std_heatmap.png").exists()
    assert (plots / "whitened_score_residual_histogram.png").exists()
