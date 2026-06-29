from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "scripts" / "audit_campaign_config_schema.py"
CONFIG = (
    ROOT
    / "examples"
    / "recipes"
    / "full_fidelity_algorithm_campaign_template"
    / "full_fidelity_info_damped_detector_ke_projected_30min_v1.yaml"
)


def _module():
    scripts_dir = str(SCRIPT.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("audit_campaign_config_schema_test", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_config(tmp_path: Path, windows: int, subblocks_per_window: int) -> Path:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    exp = cfg["experiment"]
    exp["run_name"] = f"cadence_{windows}x{subblocks_per_window}"
    exp["iterative"]["windows_per_draw"] = windows
    exp["iterative"]["subblocks_per_window"] = subblocks_per_window
    exp["iterative_forecast"].pop("actual_windows", None)
    exp["iterative_forecast"].pop("subblocks_per_window", None)
    exp["subblocks"].pop("n_subblocks", None)
    exp["subblocks"]["trace_source"]["window"].pop("n_subblocks", None)
    path = tmp_path / f"cadence_{windows}x{subblocks_per_window}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def test_audit_base_projected_30min_config() -> None:
    module = _module()
    audit = module.build_audit(CONFIG)

    assert audit["errors"] == []
    assert audit["experiment_kind"] == "full_fidelity_binary_iterative"
    assert audit["windows_per_draw"] == 10
    assert audit["subblocks_per_window"] == 30
    assert audit["total_realized_subblocks"] == 300
    assert audit["iterative_forecast"]["projected_windows"] == 60


def test_audit_accepts_common_cadence_variants(tmp_path: Path) -> None:
    module = _module()

    for windows, subblocks_per_window in ((10, 30), (5, 60), (3, 100)):
        audit = module.build_audit(_write_config(tmp_path, windows, subblocks_per_window))
        assert audit["errors"] == []
        assert audit["total_realized_subblocks"] == 300
        assert audit["iterative_forecast"]["actual_windows"] == windows
        assert audit["iterative_forecast"]["subblocks_per_window"] == subblocks_per_window


def test_audit_rejects_stale_iterative_forecast_actual_windows(tmp_path: Path) -> None:
    module = _module()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    exp = cfg["experiment"]
    exp["iterative"]["windows_per_draw"] = 5
    exp["iterative"]["subblocks_per_window"] = 60
    exp["iterative_forecast"]["actual_windows"] = 10
    exp["iterative_forecast"]["subblocks_per_window"] = 60
    path = tmp_path / "stale_forecast.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    audit = module.build_audit(path)

    assert any("actual_windows conflicts" in error for error in audit["errors"])


def test_audit_rejects_stale_subblock_count(tmp_path: Path) -> None:
    module = _module()
    path = _write_config(tmp_path, 5, 60)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["experiment"]["subblocks"]["n_subblocks"] = 600
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    audit = module.build_audit(path)

    assert any("subblocks.n_subblocks conflicts" in error for error in audit["errors"])


def test_audit_manifest_detects_doubled_results_root(tmp_path: Path) -> None:
    module = _module()
    config_path = _write_config(tmp_path, 5, 60)
    manifest_path = tmp_path / "shard_manifest.csv"
    row = {
        "shard_name": "cadence_cond_a_draw_000",
        "shard_mode": "draw",
        "source_config_path": str(config_path),
        "config_path": str(config_path),
        "expected_run_root": (
            "/projects/results/observation_bias_campaign/"
            "observation_bias_campaign/cadence_cond_a_draw_000"
        ),
        "condition_label": "a",
        "draw_start": "0",
        "draw_stop": "1",
        "draw_index": "0",
        "expected_subblocks": "300",
        "expected_windows": "5",
        "expected_subblocks_per_window": "60",
        "expected_n_theta": "20",
        "recommended_time": "12:00:00",
        "recommended_cpus_per_task": "10",
        "recommended_mem": "128G",
        "recommended_max_workers": "5",
        "sbatch_command": (
            "sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G "
            "--export=ALL,MAX_WORKERS=5 wrapper.sbatch"
        ),
    }
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    audit = module.build_audit(config_path, manifest_path)

    assert audit["shard_manifest"]["row_count"] == 1
    assert any("doubled" in error for error in audit["errors"])
