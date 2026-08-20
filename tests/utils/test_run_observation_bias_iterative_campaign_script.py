from __future__ import annotations

import importlib.util
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from dluxshera.utils.iterative_campaigns import apply_physical_reference_update


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "scripts"
    / "run_observation_bias_campaign.py"
)


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "run_observation_bias_campaign_iterative_test",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_iterative_config(path: Path) -> None:
    payload = {
        "experiment": {
            "kind": "observation_bias_campaign",
            "run_name": "binary_iterative_unit",
            "subblocks": {
                "n_frames": 3,
                "noise": "disabled",
                "phi_ref": "truth_when_available",
                "schur_curvature_method": "auto",
                "max_dense_dim": 40,
                "schur_damping": 1.0e-8,
                "summary_information_scale": "summed_likelihood",
                "trace_source": {"mode": "iid_jitter"},
            },
            "iterative": {
                "enabled": True,
                "windows_per_draw": 2,
                "subblocks_per_window": 1,
                "update_gain": 1.0,
                "update_mode": "physical_full",
                "carry_prior_mean_with_reference": True,
            },
            "seeding": {
                "seed_policy": "different_jitter_different_noise",
                "base_seed": 42,
            },
            "observation_theta": {
                "source": {
                    "separation_as": True,
                    "log_flux_total": False,
                    "contrast": False,
                },
                "optics": {
                    "plate_scale_as_per_pix": False,
                    "primary_zernikes": {
                        "enabled": True,
                        "indices": "from_system",
                        "include": [0],
                        "exclude": [],
                    },
                    "secondary_zernikes": {
                        "enabled": False,
                        "indices": "from_system",
                        "include": [],
                        "exclude": [],
                    },
                },
            },
            "bias_cases": [
                {
                    "case_name": "sep_bias",
                    "theta_reference_offsets": {"source.separation_as": 2.0e-6},
                }
            ],
            "case_generation": {"include_implicit_zero_bias": False},
            "prior_draws": {"enabled": False},
            "forecast": {"enabled": False, "plots": False},
            "eigenbasis": {"enabled": False},
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_iterative_config_variant(
    path: Path,
    *,
    windows_per_draw: int = 2,
    subblocks_per_window: int = 1,
    render_retention: str | None = None,
) -> None:
    write_iterative_config(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["experiment"]["iterative"]["windows_per_draw"] = int(windows_per_draw)
    payload["experiment"]["iterative"]["subblocks_per_window"] = int(subblocks_per_window)
    if render_retention is not None:
        payload["experiment"]["subblocks"]["render_retention"] = render_retention
    path.write_text(json.dumps(payload), encoding="utf-8")


def module_rows_for_window(plan: Any, case_name: str, window_index: int) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in plan.iterative_plan_rows
        if row["case_name"] == case_name and int(row["window_index"]) == int(window_index)
    ]


def write_render_fixture(summary_path: Path, *, size: int = 11) -> dict[str, Path]:
    subblock_root = summary_path.parent.parent.parent
    render_dir = subblock_root / "render"
    render_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "cube": render_dir / "synthetic_20260101-000000_cube.fits",
        "variance": render_dir / "synthetic_20260101-000000_variance.fits",
        "manifest": render_dir / "manifest.json",
        "frame_truth": render_dir / "synthetic_20260101-000000_frame_truth.csv",
        "diagnostic": render_dir / "diagnostic.fits",
        "trace_log": subblock_root / "subprocess.stdout.log",
    }
    paths["cube"].write_bytes(b"c" * size)
    paths["variance"].write_bytes(b"v" * (size + 1))
    paths["manifest"].write_text(
        json.dumps(
            {
                "artifacts": {
                    "cube_fits": paths["cube"].name,
                    "variance_fits": paths["variance"].name,
                    "frame_truth_csv": paths["frame_truth"].name,
                }
            }
        ),
        encoding="utf-8",
    )
    paths["frame_truth"].write_text("frame_index\n0\n", encoding="utf-8")
    paths["diagnostic"].write_bytes(b"keep")
    paths["trace_log"].write_text("keep\n", encoding="utf-8")
    return paths


def write_synthetic_subblock_summary(summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    (summary_path.parent / "subblock_summary_matrices.npz").write_bytes(b"matrix")
    (summary_path.parent / "schur_diagnostics.json").write_text("{}", encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "image_backed_subblock_summary.v1",
                "theta_labels": ["source.separation_as"],
                "theta_ref": [0.0],
                "reduced_information": [[1.0]],
                "reduced_score": [0.0],
                "information_accounting": {
                    "summary_information_scale": "summed_likelihood"
                },
            }
        ),
        encoding="utf-8",
    )


def write_completed_window_artifacts(module: Any, plan: Any, row: dict[str, Any]) -> None:
    posterior_path = Path(row["case_posterior_path"])
    window_root = posterior_path.parent
    window_root.mkdir(parents=True, exist_ok=True)
    truth_sep = float(plan.prior_truth[plan.layout.labels.index("source.separation_as")])
    posterior_lines = [
        "case_name,theta_label,truth_value,reference_value,posterior_mean,posterior_sigma"
    ]
    for label in plan.layout.labels:
        truth = truth_sep if label == "source.separation_as" else 0.0
        posterior_lines.append(
            f"{row['case_name']},{label},{truth},0.0,{truth},1e-7"
        )
    posterior_path.write_text("\n".join(posterior_lines) + "\n", encoding="utf-8")
    Path(row["window_summary_path"]).write_text("case_name\nsynthetic\n", encoding="utf-8")
    Path(row["window_diagnostic_path"]).write_text(
        "case_name,window_index\nsynthetic,0\n",
        encoding="utf-8",
    )
    module._write_json(
        Path(row["iterative_reference_update_path"]),
        {
            "schema_version": "observation_bias_iterative_reference_update.v1",
            "window_index": int(row["window_index"]),
            "status": "ok",
            "posterior_table_path": str(posterior_path),
        },
    )


def render_fits_for_rows(rows: list[dict[str, Any]]) -> list[Path]:
    out: list[Path] = []
    for row in rows:
        subblock_root = Path(row["summary_path"]).parent.parent.parent
        out.extend(sorted((subblock_root / "render").glob("*_cube.fits")))
        out.extend(sorted((subblock_root / "render").glob("*_variance.fits")))
    return out


def test_apply_physical_reference_update_ignores_missing_and_nonfinite() -> None:
    current = {"a": 2.0, "b": 5.0}
    posterior = {
        "a": {"theta_label": "a", "posterior_mean": "13.0"},
        "b": {"theta_label": "b", "posterior_mean": "nan"},
        "c": {"theta_label": "c", "posterior_mean": "99.0"},
    }
    truth = {"a": 10.0, "b": 100.0}

    updated = apply_physical_reference_update(
        current_offsets=current,
        posterior_rows_by_label=posterior,
        truth_by_label=truth,
        update_gain=0.5,
    )

    assert updated["a"] == pytest.approx(2.5)
    assert updated["b"] == pytest.approx(5.0)
    assert "c" not in updated


def test_binary_iterative_dry_run_writes_stable_window_contract(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "iterative.json"
    write_iterative_config(config_path)

    payload = module.run_observation_bias_campaign(
        config_path=config_path,
        results_root=tmp_path,
        run_name="iterative_plan",
        dry_run=True,
        system_preset="SHERA_FLIGHT_3P",
        quiet=True,
    )
    run_root = tmp_path / "iterative_plan"

    assert payload["iterative"]["enabled"] is True
    assert payload["iterative"]["windows_per_draw"] == 2
    assert (run_root / "campaign_plan.json").exists()
    assert (run_root / "iterative_plan.csv").exists()
    assert (run_root / "expected_outputs.csv").exists()
    expected = payload["expected_outputs"]
    assert len(expected) == 2
    assert {row["window_index"] for row in expected} == {0, 1}
    assert all("case_posterior_path" in row for row in expected)
    assert all("summary_path" in row for row in expected)
    assert all("window_case_name" in row for row in expected)
    assert all("window_summary_path" in row for row in expected)
    assert all("iterative_reference_update_path" in row for row in expected)
    assert all("realized_command_path" in row for row in expected)
    first_iter = payload["iterative_plan"][0]
    assert first_iter["theta_reference_offsets_window0_json"]
    assert first_iter["trace_source_mode"] == "iid_jitter"
    assert first_iter["realized_after_reference_update"] is True


def test_render_retention_cli_override_resolves_in_plan(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "iterative.json"
    write_iterative_config_variant(config_path, windows_per_draw=1, subblocks_per_window=1)

    payload = module.run_observation_bias_campaign(
        config_path=config_path,
        results_root=tmp_path,
        run_name="retention_cli",
        dry_run=True,
        system_preset="SHERA_FLIGHT_3P",
        quiet=True,
        args=module.argparse.Namespace(
            aggregate_only=False,
            resume=False,
            run_name="retention_cli",
            n_subblocks=None,
            n_frames=None,
            trace_source_mode=None,
            trajectory_csv=None,
            trajectory_start_s=None,
            trajectory_duration_s=None,
            trajectory_n_subblocks=None,
            trajectory_frame_dt_s=None,
            trajectory_output_keys=None,
            trajectory_plan=None,
            noise=None,
            phi_ref=None,
            max_dense_dim=None,
            schur_curvature_method=None,
            summary_information_scale=None,
            render_retention="delete_after_window",
            profile_runtime=False,
            profile_runtime_detail=None,
            memory_diagnostics=False,
            seed_policy=None,
            base_seed=None,
        ),
    )

    assert payload["iterative"]["render_retention"] == "delete_after_window"


def test_render_retention_default_keep_preserves_rendered_fits(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "iterative.json"
    write_iterative_config_variant(config_path, windows_per_draw=1, subblocks_per_window=2)
    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="retention_default",
        system_preset="SHERA_FLIGHT_3P",
    )
    case = plan.cases[0]
    rows = module_rows_for_window(plan, case.case_name, 0)
    for row in rows:
        write_synthetic_subblock_summary(Path(row["summary_path"]))
        write_render_fixture(Path(row["summary_path"]))
    write_completed_window_artifacts(module, plan, rows[0])

    result = module._cleanup_completed_iterative_window_renders(
        plan=plan,
        window_case=module.BiasCase(
            case_name=rows[0]["window_case_name"],
            theta_reference_offsets={},
        ),
        window_index=0,
        posterior_path=Path(rows[0]["case_posterior_path"]),
        summary_paths=[Path(row["summary_path"]) for row in rows],
    )

    assert result["cleanup_status"] == "skipped_policy_keep"
    assert len(render_fits_for_rows(rows)) == 4
    assert not (
        tmp_path
        / "retention_default"
        / "cases"
        / rows[0]["window_case_name"]
        / "render_retention"
        / "cleanup_latest.json"
    ).exists()


def test_render_retention_explicit_keep_preserves_rendered_fits(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "iterative.json"
    write_iterative_config_variant(
        config_path,
        windows_per_draw=1,
        subblocks_per_window=1,
        render_retention="keep",
    )
    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="retention_keep",
        system_preset="SHERA_FLIGHT_3P",
    )
    case = plan.cases[0]
    rows = module_rows_for_window(plan, case.case_name, 0)
    write_synthetic_subblock_summary(Path(rows[0]["summary_path"]))
    write_render_fixture(Path(rows[0]["summary_path"]))
    write_completed_window_artifacts(module, plan, rows[0])

    result = module._cleanup_completed_iterative_window_renders(
        plan=plan,
        window_case=module.BiasCase(
            case_name=rows[0]["window_case_name"],
            theta_reference_offsets={},
        ),
        window_index=0,
        posterior_path=Path(rows[0]["case_posterior_path"]),
        summary_paths=[Path(rows[0]["summary_path"])],
    )

    assert result["cleanup_status"] == "skipped_policy_keep"
    assert len(render_fits_for_rows(rows)) == 2


def test_delete_after_window_removes_only_transient_fits_after_guard(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "iterative.json"
    write_iterative_config_variant(
        config_path,
        windows_per_draw=1,
        subblocks_per_window=2,
        render_retention="delete_after_window",
    )
    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="retention_delete",
        system_preset="SHERA_FLIGHT_3P",
    )
    case = plan.cases[0]
    rows = module_rows_for_window(plan, case.case_name, 0)
    persistent_paths: list[Path] = []
    for row in rows:
        write_synthetic_subblock_summary(Path(row["summary_path"]))
        artifacts = write_render_fixture(Path(row["summary_path"]), size=7)
        persistent_paths.extend(
            [
                Path(row["summary_path"]),
                Path(row["summary_path"]).parent / "subblock_summary_matrices.npz",
                Path(row["summary_path"]).parent / "schur_diagnostics.json",
                artifacts["manifest"],
                artifacts["frame_truth"],
                artifacts["diagnostic"],
                artifacts["trace_log"],
            ]
        )
    write_completed_window_artifacts(module, plan, rows[0])

    result = module._cleanup_completed_iterative_window_renders(
        plan=plan,
        window_case=module.BiasCase(
            case_name=rows[0]["window_case_name"],
            theta_reference_offsets={},
        ),
        window_index=0,
        posterior_path=Path(rows[0]["case_posterior_path"]),
        summary_paths=[Path(row["summary_path"]) for row in rows],
    )

    assert result["cleanup_status"] == "ok"
    assert result["files_deleted_count"] == 4
    assert result["logical_bytes_deleted"] == 30
    assert render_fits_for_rows(rows) == []
    assert all(path.exists() for path in persistent_paths)
    provenance = json.loads(
        (
            tmp_path
            / "retention_delete"
            / "cases"
            / rows[0]["window_case_name"]
            / "render_retention"
            / "cleanup_latest.json"
        ).read_text(encoding="utf-8")
    )
    assert provenance["retention_policy"] == "delete_after_window"
    assert provenance["completion_guard"]["status"] == "ok"
    assert "posterior_by_label_csv" in provenance["completion_guard"][
        "canonical_completed_window_artifacts"
    ]


def test_delete_after_window_is_idempotent(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "iterative.json"
    write_iterative_config_variant(
        config_path,
        windows_per_draw=1,
        subblocks_per_window=1,
        render_retention="delete_after_window",
    )
    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="retention_idempotent",
        system_preset="SHERA_FLIGHT_3P",
    )
    case = plan.cases[0]
    rows = module_rows_for_window(plan, case.case_name, 0)
    write_synthetic_subblock_summary(Path(rows[0]["summary_path"]))
    write_render_fixture(Path(rows[0]["summary_path"]))
    write_completed_window_artifacts(module, plan, rows[0])
    window_case = module.BiasCase(
        case_name=rows[0]["window_case_name"],
        theta_reference_offsets={},
    )

    first = module._cleanup_completed_iterative_window_renders(
        plan=plan,
        window_case=window_case,
        window_index=0,
        posterior_path=Path(rows[0]["case_posterior_path"]),
        summary_paths=[Path(rows[0]["summary_path"])],
    )
    second = module._cleanup_completed_iterative_window_renders(
        plan=plan,
        window_case=window_case,
        window_index=0,
        posterior_path=Path(rows[0]["case_posterior_path"]),
        summary_paths=[Path(rows[0]["summary_path"])],
    )

    assert first["files_deleted_count"] == 2
    assert second["cleanup_status"] == "ok"
    assert second["files_deleted_count"] == 0
    history = (
        tmp_path
        / "retention_idempotent"
        / "cases"
        / rows[0]["window_case_name"]
        / "render_retention"
        / "cleanup_history.jsonl"
    ).read_text(encoding="utf-8")
    assert len([line for line in history.splitlines() if line.strip()]) == 2


def test_delete_after_window_guard_failure_deletes_nothing(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "iterative.json"
    write_iterative_config_variant(
        config_path,
        windows_per_draw=1,
        subblocks_per_window=1,
        render_retention="delete_after_window",
    )
    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="retention_guard",
        system_preset="SHERA_FLIGHT_3P",
    )
    case = plan.cases[0]
    rows = module_rows_for_window(plan, case.case_name, 0)
    write_synthetic_subblock_summary(Path(rows[0]["summary_path"]))
    write_render_fixture(Path(rows[0]["summary_path"]))

    result = module._cleanup_completed_iterative_window_renders(
        plan=plan,
        window_case=module.BiasCase(
            case_name=rows[0]["window_case_name"],
            theta_reference_offsets={},
        ),
        window_index=0,
        posterior_path=Path(rows[0]["case_posterior_path"]),
        summary_paths=[Path(rows[0]["summary_path"])],
    )

    assert result["cleanup_status"] == "guard_failed"
    assert result["files_deleted_count"] == 0
    assert len(render_fits_for_rows(rows)) == 2


def test_delete_after_window_records_unlink_failure_without_raising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module()
    config_path = tmp_path / "iterative.json"
    write_iterative_config_variant(
        config_path,
        windows_per_draw=1,
        subblocks_per_window=1,
        render_retention="delete_after_window",
    )
    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="retention_failure",
        system_preset="SHERA_FLIGHT_3P",
    )
    case = plan.cases[0]
    rows = module_rows_for_window(plan, case.case_name, 0)
    write_synthetic_subblock_summary(Path(rows[0]["summary_path"]))
    write_render_fixture(Path(rows[0]["summary_path"]))
    write_completed_window_artifacts(module, plan, rows[0])
    original_unlink = module.Path.unlink

    def flaky_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
        if self.name.endswith("_variance.fits"):
            raise PermissionError("synthetic permission failure")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(module.Path, "unlink", flaky_unlink)
    result = module._cleanup_completed_iterative_window_renders(
        plan=plan,
        window_case=module.BiasCase(
            case_name=rows[0]["window_case_name"],
            theta_reference_offsets={},
        ),
        window_index=0,
        posterior_path=Path(rows[0]["case_posterior_path"]),
        summary_paths=[Path(rows[0]["summary_path"])],
    )

    assert result["cleanup_status"] == "failed"
    assert result["files_deleted_count"] == 1
    assert result["errors"][0]["error"] == "synthetic permission failure"
    assert Path(rows[0]["case_posterior_path"]).exists()


def test_partial_window_renders_preserved_until_resume_completes_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module()
    config_path = tmp_path / "iterative.json"
    write_iterative_config_variant(
        config_path,
        windows_per_draw=1,
        subblocks_per_window=2,
        render_retention="delete_after_window",
    )
    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="partial_resume",
        system_preset="SHERA_FLIGHT_3P",
    )
    case = plan.cases[0]
    rows = module_rows_for_window(plan, case.case_name, 0)
    write_synthetic_subblock_summary(Path(rows[0]["summary_path"]))
    write_render_fixture(Path(rows[0]["summary_path"]))
    executed: list[str] = []

    def fake_execute_subblocks(window_plan_obj: Any, **kwargs: Any) -> None:
        assert kwargs["resume"] is True
        assert len(render_fits_for_rows([rows[0]])) == 2
        status_rows = []
        for summary_path in window_plan_obj.summary_paths[window_plan_obj.cases[0].case_name]:
            if Path(summary_path).exists():
                continue
            executed.append(str(summary_path))
            write_synthetic_subblock_summary(Path(summary_path))
            write_render_fixture(Path(summary_path))
            status_rows.append(
                {
                    "case_name": window_plan_obj.cases[0].case_name,
                    "summary_path": str(summary_path),
                    "status": "ok",
                    "return_code": 0,
                }
            )
        module._write_csv_rows(window_plan_obj.run_root / "subblock_status.csv", status_rows)

    def fake_aggregate_case(*, plan: Any, case: Any, **kwargs: Any) -> dict[str, Any]:
        row = plan.subblock_plans[case.case_name][0]
        write_completed_window_artifacts(module, plan, row)
        return {"case_name": case.case_name, "case_root": str(Path(row["case_posterior_path"]).parent)}

    monkeypatch.setattr(module, "execute_subblocks", fake_execute_subblocks)
    monkeypatch.setattr(module, "aggregate_case", fake_aggregate_case)

    status = module.execute_iterative_campaign(
        plan,
        resume=True,
        max_workers=1,
        fail_fast=True,
        quiet=True,
        prior_source="summary_theta_ref",
        allow_optimizer_scale_summaries=False,
        resource_time="disabled",
    )

    assert executed == [rows[1]["summary_path"]]
    assert render_fits_for_rows(rows) == []
    assert status["missing_posterior_tables"] == 0


def test_completed_and_already_pruned_windows_resume_without_recompute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module()
    config_path = tmp_path / "iterative.json"
    write_iterative_config_variant(
        config_path,
        windows_per_draw=2,
        subblocks_per_window=1,
        render_retention="delete_after_window",
    )
    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="completed_resume",
        system_preset="SHERA_FLIGHT_3P",
    )
    case = plan.cases[0]
    rows0 = module_rows_for_window(plan, case.case_name, 0)
    rows1 = module_rows_for_window(plan, case.case_name, 1)
    for row in rows0 + rows1:
        write_synthetic_subblock_summary(Path(row["summary_path"]))
        write_completed_window_artifacts(module, plan, row)
    write_render_fixture(Path(rows0[0]["summary_path"]))

    def fail_execute(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("completed windows should not execute subblocks")

    def fail_aggregate(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("completed windows should not re-aggregate")

    monkeypatch.setattr(module, "execute_subblocks", fail_execute)
    monkeypatch.setattr(module, "aggregate_case", fail_aggregate)

    status = module.execute_iterative_campaign(
        plan,
        resume=True,
        max_workers=1,
        fail_fast=True,
        quiet=True,
        prior_source="summary_theta_ref",
        allow_optimizer_scale_summaries=False,
        resource_time="disabled",
    )

    assert render_fits_for_rows(rows0) == []
    assert render_fits_for_rows(rows1) == []
    assert status["missing_posterior_tables"] == 0


def test_mixed_campaign_cleans_completed_preserves_current_until_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module()
    config_path = tmp_path / "iterative.json"
    write_iterative_config_variant(
        config_path,
        windows_per_draw=3,
        subblocks_per_window=2,
        render_retention="delete_after_window",
    )
    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="mixed_resume",
        system_preset="SHERA_FLIGHT_3P",
    )
    case = plan.cases[0]
    rows0 = module_rows_for_window(plan, case.case_name, 0)
    rows1 = module_rows_for_window(plan, case.case_name, 1)
    rows2 = module_rows_for_window(plan, case.case_name, 2)
    for row in rows0:
        write_synthetic_subblock_summary(Path(row["summary_path"]))
        write_render_fixture(Path(row["summary_path"]))
    write_completed_window_artifacts(module, plan, rows0[0])
    write_synthetic_subblock_summary(Path(rows1[0]["summary_path"]))
    write_render_fixture(Path(rows1[0]["summary_path"]))
    executed: list[str] = []
    preserved_partial_seen = False

    def fake_execute_subblocks(window_plan_obj: Any, **kwargs: Any) -> None:
        nonlocal preserved_partial_seen
        window_rows = list(window_plan_obj.subblock_plans[window_plan_obj.cases[0].case_name])
        if int(window_rows[0]["window_index"]) == 1:
            preserved_partial_seen = len(render_fits_for_rows([rows1[0]])) == 2
        status_rows = []
        for summary_path in window_plan_obj.summary_paths[window_plan_obj.cases[0].case_name]:
            if Path(summary_path).exists():
                continue
            executed.append(str(summary_path))
            write_synthetic_subblock_summary(Path(summary_path))
            write_render_fixture(Path(summary_path))
            status_rows.append({"summary_path": str(summary_path), "status": "ok", "return_code": 0})
        module._write_csv_rows(window_plan_obj.run_root / "subblock_status.csv", status_rows)

    def fake_aggregate_case(*, plan: Any, case: Any, **kwargs: Any) -> dict[str, Any]:
        row = plan.subblock_plans[case.case_name][0]
        write_completed_window_artifacts(module, plan, row)
        return {"case_name": case.case_name, "case_root": str(Path(row["case_posterior_path"]).parent)}

    monkeypatch.setattr(module, "execute_subblocks", fake_execute_subblocks)
    monkeypatch.setattr(module, "aggregate_case", fake_aggregate_case)

    module.execute_iterative_campaign(
        plan,
        resume=True,
        max_workers=1,
        fail_fast=True,
        quiet=True,
        prior_source="summary_theta_ref",
        allow_optimizer_scale_summaries=False,
        resource_time="disabled",
    )

    assert preserved_partial_seen is True
    assert rows1[1]["summary_path"] in executed
    assert rows2[0]["summary_path"] in executed
    assert rows2[1]["summary_path"] in executed
    assert render_fits_for_rows(rows0) == []
    assert render_fits_for_rows(rows1) == []
    assert render_fits_for_rows(rows2) == []


def test_binary_iterative_aggregate_only_reports_missing_outputs(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "iterative.json"
    write_iterative_config(config_path)
    module.run_observation_bias_campaign(
        config_path=config_path,
        results_root=tmp_path,
        run_name="missing_plan",
        dry_run=True,
        system_preset="SHERA_FLIGHT_3P",
        quiet=True,
    )

    status = module.run_observation_bias_campaign(
        config_path=config_path,
        results_root=tmp_path,
        run_name="missing_plan",
        aggregate_only=True,
        system_preset="SHERA_FLIGHT_3P",
        quiet=True,
    )

    analysis = tmp_path / "missing_plan" / "analysis"
    assert status["missing_summaries"] == 2
    assert status["missing_posterior_tables"] == 2
    assert status["missing_outputs_by_kind"]["iterative_reference_update"] == 2
    assert (analysis / "missing_outputs.csv").exists()
    assert (analysis / "output_inventory.csv").exists()
    assert (analysis / "aggregate_status.json").exists()


def test_binary_iterative_accepts_eigen_damped_update_mode(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "iterative.json"
    write_iterative_config(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["experiment"]["iterative"]["update_mode"] = "eigen_damped"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    result = module.run_observation_bias_campaign(
        config_path=config_path,
        results_root=tmp_path,
        run_name="eigen_damped_mode",
        dry_run=True,
        system_preset="SHERA_FLIGHT_3P",
        quiet=True,
    )

    assert result["iterative"]["update_mode"] == "eigen_damped"
    assert result["iterative"]["update_policy"]["update_mode"] == "eigen_damped"
    assert "eigenbasis" in result["iterative"]
    assert result["iterative"]["eigenbasis"]["basis_source"] == "posterior_precision"


def test_binary_iterative_rejects_unknown_update_mode(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "iterative.json"
    write_iterative_config(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["experiment"]["iterative"]["update_mode"] = "eigen_future"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="experiment.iterative.update_mode must be one of"):
        module.run_observation_bias_campaign(
            config_path=config_path,
            results_root=tmp_path,
            run_name="bad_mode",
            dry_run=True,
            system_preset="SHERA_FLIGHT_3P",
            quiet=True,
        )


def test_binary_iterative_aggregate_only_reconstructs_synthetic_diagnostics(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "iterative.json"
    write_iterative_config(config_path)
    payload = module.run_observation_bias_campaign(
        config_path=config_path,
        results_root=tmp_path,
        run_name="synthetic_iterative",
        dry_run=True,
        system_preset="SHERA_FLIGHT_3P",
        quiet=True,
    )
    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="synthetic_iterative",
        system_preset="SHERA_FLIGHT_3P",
    )
    truth_sep = float(plan.prior_truth[plan.layout.labels.index("source.separation_as")])

    posterior_offsets = [1.0e-6, 0.25e-6]
    for row, sep_offset in zip(payload["expected_outputs"], posterior_offsets, strict=True):
        posterior_path = Path(row["case_posterior_path"])
        posterior_path.parent.mkdir(parents=True, exist_ok=True)
        posterior_path.write_text(
            "case_name,theta_label,truth_value,reference_value,posterior_mean,posterior_sigma\n"
            f"sep_bias,source.separation_as,{truth_sep},0.0,{truth_sep + sep_offset},1e-7\n"
            "sep_bias,optics.primary.zernike_coeffs_nm[0],0.0,0.0,0.0,1.0\n",
            encoding="utf-8",
        )
        Path(row["summary_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(row["summary_path"]).write_text("{}", encoding="utf-8")
        Path(row["window_summary_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(row["window_summary_path"]).write_text("case_name\nsep_bias\n", encoding="utf-8")

    status = module.run_observation_bias_campaign(
        config_path=config_path,
        results_root=tmp_path,
        run_name="synthetic_iterative",
        aggregate_only=True,
        system_preset="SHERA_FLIGHT_3P",
        quiet=True,
    )

    diag_path = tmp_path / "synthetic_iterative" / "analysis" / "iterative_window_diagnostics.csv"
    text = diag_path.read_text(encoding="utf-8")
    with diag_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert status["iterative_window_diagnostic_rows"] == 2
    assert "separation_posterior_update_microas" in text
    assert "separation_applied_update_microas" in text
    assert "residual_norm_decreased_from_previous_window" in text
    assert rows[0]["separation_next_reference_improved"] == "True"
    assert rows[1]["residual_norm_decreased_from_previous_window"] == "True"


def test_binary_iterative_synthetic_wrong_direction_flags_not_improved(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "iterative.json"
    write_iterative_config(config_path)
    payload = module.run_observation_bias_campaign(
        config_path=config_path,
        results_root=tmp_path,
        run_name="synthetic_wrong_direction",
        dry_run=True,
        system_preset="SHERA_FLIGHT_3P",
        quiet=True,
    )
    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="synthetic_wrong_direction",
        system_preset="SHERA_FLIGHT_3P",
    )
    truth_sep = float(plan.prior_truth[plan.layout.labels.index("source.separation_as")])

    for row in payload["expected_outputs"]:
        posterior_path = Path(row["case_posterior_path"])
        posterior_path.parent.mkdir(parents=True, exist_ok=True)
        posterior_path.write_text(
            "case_name,theta_label,truth_value,reference_value,posterior_mean,posterior_sigma\n"
            f"sep_bias,source.separation_as,{truth_sep},0.0,{truth_sep + 3.0e-6},1e-7\n"
            "sep_bias,optics.primary.zernike_coeffs_nm[0],0.0,0.0,0.0,1.0\n",
            encoding="utf-8",
        )
        Path(row["summary_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(row["summary_path"]).write_text("{}", encoding="utf-8")
        Path(row["window_summary_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(row["window_summary_path"]).write_text("case_name\nsep_bias\n", encoding="utf-8")

    module.run_observation_bias_campaign(
        config_path=config_path,
        results_root=tmp_path,
        run_name="synthetic_wrong_direction",
        aggregate_only=True,
        system_preset="SHERA_FLIGHT_3P",
        quiet=True,
    )

    diag_path = tmp_path / "synthetic_wrong_direction" / "analysis" / "iterative_window_diagnostics.csv"
    with diag_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["separation_next_reference_improved"] == "False"
    assert rows[0]["separation_update_sign_toward_truth"] == "False"
