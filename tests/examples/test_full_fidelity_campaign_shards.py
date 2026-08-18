from __future__ import annotations

import copy
import csv
import importlib.util
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "examples" / "scripts"
CONFIG = (
    ROOT
    / "examples"
    / "recipes"
    / "full_fidelity_algorithm_campaign_template"
    / "full_fidelity_zernike_2x2_self_correction_hpc_v1.yaml"
)


def _load(name: str, path: Path):
    scripts_dir = str(SCRIPTS)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _without_shard_selection(config: dict) -> dict:
    payload = copy.deepcopy(config)
    experiment = payload["experiment"]
    experiment.pop("run_name", None)
    prior = experiment["prior_draws"]
    for key in (
        "n_cases",
        "conditions",
        "condition_index_start",
        "draw_index_start",
        "global_draw_index_start",
        "rng_skip_draws",
    ):
        prior.pop(key, None)
    return payload


def test_condition_and_draw_shard_shapes_preserve_science_config() -> None:
    module = _load(
        "prepare_full_fidelity_campaign_shards_test",
        SCRIPTS / "prepare_full_fidelity_campaign_shards.py",
    )
    source = _config()
    condition_shards = module.build_shards(
        source,
        run_name_prefix="full_fidelity_zernike_2x2_self_correction_hpc_v1",
        mode="condition",
    )
    draw_shards = module.build_shards(
        source,
        run_name_prefix="full_fidelity_zernike_2x2_self_correction_hpc_v1",
        mode="draw",
    )

    assert len(condition_shards) == 4
    assert {shard.expected_subblocks for shard in condition_shards} == {75}
    assert {shard.expected_windows for shard in condition_shards} == {15}
    assert len(draw_shards) == 20
    assert {shard.expected_subblocks for shard in draw_shards} == {15}
    assert {shard.expected_windows for shard in draw_shards} == {3}
    assert len({shard.name for shard in condition_shards + draw_shards}) == 24
    assert all(
        module._safe_name(shard.name, name="shard_name") == shard.name
        for shard in condition_shards + draw_shards
    )

    source_without_selection = _without_shard_selection(source)
    for shard in condition_shards + draw_shards:
        assert _without_shard_selection(shard.config) == source_without_selection


def test_shards_preserve_parent_prior_draw_realizations() -> None:
    prepare = _load(
        "prepare_full_fidelity_campaign_shards_rng_test",
        SCRIPTS / "prepare_full_fidelity_campaign_shards.py",
    )
    runner = _load(
        "run_observation_bias_campaign_shards_rng_test",
        SCRIPTS / "run_observation_bias_campaign.py",
    )
    source = _config()
    labels = [
        "source.separation_as",
        "source.log_flux_total",
        "source.contrast",
        "optics.plate_scale_as_per_pix",
        *[f"optics.primary.zernike_coeffs_nm[{index}]" for index in range(8)],
        *[f"optics.secondary.zernike_coeffs_nm[{index}]" for index in range(8)],
    ]
    truth = {label: 1.0 for label in labels}
    parent_cases, _ = runner._generate_prior_draw_cases(
        experiment_cfg=source["experiment"],
        labels=labels,
        truth_by_label=truth,
    )
    parent_offsets = {
        case.case_name: case.theta_reference_offsets for case in parent_cases
    }

    for mode in ("condition", "draw"):
        shards = prepare.build_shards(
            source,
            run_name_prefix="rng_check",
            mode=mode,
        )
        for shard in shards:
            cases, _ = runner._generate_prior_draw_cases(
                experiment_cfg=shard.config["experiment"],
                labels=labels,
                truth_by_label=truth,
            )
            for case in cases:
                assert case.theta_reference_offsets == parent_offsets[case.case_name]


def test_generated_manifest_and_helpers(tmp_path: Path) -> None:
    module = _load(
        "prepare_full_fidelity_campaign_shards_files_test",
        SCRIPTS / "prepare_full_fidelity_campaign_shards.py",
    )
    outdir = tmp_path / "shards"
    rows = module.prepare_shards(
        config_path=CONFIG,
        outdir=outdir,
        run_name_prefix="full_fidelity_zernike_2x2_self_correction_hpc_v1",
        mode="condition",
        results_root=tmp_path / "results",
        resources=module.Resources(
            time="36:00:00",
            cpus_per_task=10,
            mem="128G",
            max_workers=5,
        ),
        dry_run=False,
        overwrite=False,
    )

    manifest_rows = list(
        csv.DictReader(
            (outdir / "shard_manifest.csv").open(
                "r",
                encoding="utf-8",
                newline="",
            )
        )
    )
    assert len(rows) == len(manifest_rows) == 4
    assert len(list((outdir / "configs").glob("*.yaml"))) == 4
    assert {int(row["expected_subblocks"]) for row in manifest_rows} == {75}
    assert {row["recommended_time"] for row in manifest_rows} == {"36:00:00"}
    assert {row["recommended_cpus_per_task"] for row in manifest_rows} == {"10"}
    assert {row["recommended_mem"] for row in manifest_rows} == {"128G"}
    assert {row["recommended_max_workers"] for row in manifest_rows} == {"5"}
    assert {row["expected_n_theta"] for row in manifest_rows} == {"20"}
    for row in manifest_rows:
        command = row["sbatch_command"]
        assert "--time=36:00:00" in command
        assert "--cpus-per-task=10" in command
        assert "--mem=128G" in command
        assert "MAX_WORKERS=5" in command
        assert "FAIL_FAST=1" in command
        assert "ANALYZE_AFTER_RUN=1" in command
        assert "USE_RESOURCE_TIME=1" in command
        assert "full_fidelity_iterative_campaign_hpc.sbatch" in command

    assert (outdir / "preflight_condition_shards.sh").stat().st_mode & 0o111
    assert (outdir / "submit_condition_shards.sh").stat().st_mode & 0o111
    assert (outdir / "summarize_shard_status.sh").stat().st_mode & 0o111
    readme = (outdir / "README.md").read_text(encoding="utf-8")
    assert "Iterative windows within a draw are" in readme
    assert "sequential" in readme
    assert "GPU benchmark" in readme


def test_condition_shard_preflight_plans_75_subblocks_each(
    tmp_path: Path,
) -> None:
    prepare = _load(
        "prepare_full_fidelity_campaign_shards_preflight_test",
        SCRIPTS / "prepare_full_fidelity_campaign_shards.py",
    )
    check = _load(
        "check_full_fidelity_campaign_shards_preflight_test",
        SCRIPTS / "check_full_fidelity_campaign_shards.py",
    )
    outdir = tmp_path / "shards"
    prepare.prepare_shards(
        config_path=CONFIG,
        outdir=outdir,
        run_name_prefix="preflight_zernike",
        mode="condition",
        results_root=None,
        resources=prepare.Resources(
            time="36:00:00",
            cpus_per_task=10,
            mem="128G",
            max_workers=5,
        ),
        dry_run=False,
        overwrite=False,
    )
    preflight_root = tmp_path / "preflight"

    assert (
        check.preflight(outdir / "shard_manifest.csv", preflight_root) == 0
    )
    run_roots = list(preflight_root.glob("preflight_zernike_cond_*"))
    assert len(run_roots) == 4
    for run_root in run_roots:
        with (run_root / "subblock_plan.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            assert len(list(csv.DictReader(handle))) == 75
