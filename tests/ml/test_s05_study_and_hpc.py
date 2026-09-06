from __future__ import annotations

import errno
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import pytest

from dluxshera.datasets.schema import write_json
from dluxshera.ml import (
    PairPolicy,
    generate_frozen_pair_manifest,
    generate_split_registry,
    load_sample_catalog,
    load_study_prescription,
    resolve_study_experiment_config,
    split_registry_content_sha256,
    write_pair_manifest,
    write_split_registry,
)
from dluxshera.ml.hpc import (
    build_sbatch_command,
    parse_sbatch_job_id,
    persist_study_contract_artifacts,
    prepare_sbatch_submission,
    slurm_profile,
)
import dluxshera.ml.hpc as hpc_module
from tests.ml.test_catalog_splits_pairs import _write_prepared_fixture
from tests.ml.test_study_prescriptions import _study


ROOT = Path(__file__).resolve().parents[2]
S01_STUDY = ROOT / "work" / "experiments" / "ml" / "s01" / "study.yaml"
S05_STUDY = ROOT / "work" / "experiments" / "ml" / "s05" / "study.yaml"

EXPECTED_DATASET = {
    "artifact_id": "PREP-V3-v1",
    "prepared_dataset_hash": "4cdc325fbf8d4a0e07195ab075bea6f5035dfc01c9990cac03ee1f59c131e5e6",
}
EXPECTED_SPLIT = {
    "artifact_id": "SPLIT-ML-v1",
    "content_sha256": "29f0e95c3819cbeb5ce00aafb593445510723ea5fc20e2e7f3e585c1b9615314",
}
EXPECTED_PAIR_POLICY_CORE = {
    "policy_id": "s01_clean_same_pair_grid_v1",
    "family_weights": {"same_nuisance_different_science": 1.0},
    "same_pair_id": True,
    "min_fisher_distance": 0.0,
    "max_fisher_distance": 5000.0,
    "include_reverse": True,
    "max_sampling_attempts": 4000,
}
EXPECTED_EVAL_HASHES = {
    "validation": {
        "artifact_id": "S01-VALIDATION-PAIRS-v1",
        "content_sha256": "68ccd41a35d286c8b060f291eef6c788a6b0d97c9660868f74e01b2b4feae499",
        "ordered_pair_count": 2048,
    },
    "test": {
        "artifact_id": "S01-TEST-PAIRS-v1",
        "content_sha256": "375451064bd363a6afb33c6f3f1bdff7e92efe1384c1513b3491b42318c87b82",
        "ordered_pair_count": 4096,
    },
}
S05_PARAM_COUNTS = {
    "S05-E01": 767220,
    "S05-E02": 701684,
    "S05-E03": 193540,
    "S05-E04": 3055060,
}


def _submit_module():
    path = ROOT / "work" / "experiments" / "ml" / "hpc" / "submit_study_run.py"
    spec = importlib.util.spec_from_file_location("submit_study_run_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _flatten(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(_flatten(value, path))
        else:
            out[path] = value
    return out


def _changed_paths(left: Mapping[str, Any], right: Mapping[str, Any]) -> set[str]:
    flat_left = _flatten(left)
    flat_right = _flatten(right)
    keys = set(flat_left) | set(flat_right)
    return {key for key in keys if flat_left.get(key) != flat_right.get(key)}


def _assert_pair_policy_core(policy: Mapping[str, Any]) -> None:
    for key, value in EXPECTED_PAIR_POLICY_CORE.items():
        assert policy[key] == value


def test_s01_e01_prescription_keeps_canonical_scientific_identity() -> None:
    study = load_study_prescription(S01_STUDY)
    config = resolve_study_experiment_config(study, experiment_id="S01-E01")
    assert study["dataset"] == EXPECTED_DATASET
    assert study["split_registry"] == EXPECTED_SPLIT
    _assert_pair_policy_core(config["pair_policy"])
    assert config["run_id"] == "S01-E01-R001"
    assert config["seed"] == 11
    assert config["model"] == {
        "channels": [16, 32, 64, 128],
        "embedding_dim": 128,
        "encoder_hidden_dim": 256,
        "head_hidden_dim": 256,
        "comparator": "concat_diff",
        "normalization": "batch",
        "adaptive_pool_shape": [4, 4],
    }
    assert config["training"]["learning_rate"] == 0.0005
    assert config["training"]["optimizer"] == "adamw"
    assert config["evaluate_test"] is False


def test_s05_reuses_s01_benchmark_artifact_identities() -> None:
    study = load_study_prescription(S05_STUDY)
    assert study["dataset"] == EXPECTED_DATASET
    assert study["split_registry"] == EXPECTED_SPLIT
    for artifact_key, expected in EXPECTED_EVAL_HASHES.items():
        artifact = study["evaluation_artifacts"][artifact_key]
        for key, value in expected.items():
            assert artifact[key] == value
        assert artifact["pair_policy_id"] == "s01_clean_same_pair_grid_v1"
    for experiment_id in ("S05-E01", "S05-E02", "S05-E03", "S05-E04"):
        config = resolve_study_experiment_config(study, experiment_id=experiment_id)
        _assert_pair_policy_core(config["pair_policy"])
        assert config["dataset"] == EXPECTED_DATASET
        assert config["seed"] == 11
        assert config["evaluate_test"] is False


def test_s05_e01_matches_s01_baseline_except_identity_fields() -> None:
    s01 = resolve_study_experiment_config(
        load_study_prescription(S01_STUDY),
        experiment_id="S01-E01",
    )
    s05 = resolve_study_experiment_config(
        load_study_prescription(S05_STUDY),
        experiment_id="S05-E01",
    )
    assert _changed_paths(s01, s05) == {"study_id", "experiment_id", "run_id"}


def test_s05_wave1_changes_only_intended_architecture_fields() -> None:
    study = load_study_prescription(S05_STUDY)
    baseline = resolve_study_experiment_config(study, experiment_id="S05-E01")
    expected_changes = {
        "S05-E02": {"experiment_id", "run_id", "model.comparator"},
        "S05-E03": {
            "experiment_id",
            "run_id",
            "model.channels",
            "model.embedding_dim",
            "model.encoder_hidden_dim",
            "model.head_hidden_dim",
        },
        "S05-E04": {
            "experiment_id",
            "run_id",
            "model.channels",
            "model.embedding_dim",
            "model.encoder_hidden_dim",
            "model.head_hidden_dim",
        },
    }
    for experiment_id, changes in expected_changes.items():
        config = resolve_study_experiment_config(study, experiment_id=experiment_id)
        assert _changed_paths(baseline, config) == changes


def test_s05_models_instantiate_emit_twenty_outputs_and_have_stable_counts() -> None:
    torch = pytest.importorskip("torch")
    from dluxshera.ml.models import build_pairwise_correction_model, count_parameters

    study = load_study_prescription(S05_STUDY)
    image_a = torch.randn(2, 1, 64, 64)
    image_b = torch.randn(2, 1, 64, 64)
    expected_head_inputs = {
        "S05-E01": 384,
        "S05-E02": 128,
        "S05-E03": 192,
        "S05-E04": 768,
    }
    for experiment_id, parameter_count in S05_PARAM_COUNTS.items():
        config = resolve_study_experiment_config(study, experiment_id=experiment_id)
        model = build_pairwise_correction_model(20, config["model"])
        assert model(image_a, image_b).shape == (2, 20)
        assert model.regression_head[0].in_features == expected_head_inputs[experiment_id]
        assert model.regression_head[-1].out_features == 20
        assert count_parameters(model) == parameter_count


def test_slurm_profiles_construct_expected_site_commands() -> None:
    ls6 = slurm_profile("tacc_ls6")
    ls6_cmd = build_sbatch_command(
        ls6,
        script=Path("work/experiments/ml/hpc/sites/tacc_ls6/train_ml.sbatch"),
        job_name="S05-E01-R001",
    )
    assert "--partition=gpu-a100-small" in ls6_cmd
    assert "--account=JPL-PUB" in ls6_cmd
    assert "--cpus-per-task=8" in ls6_cmd
    assert not any(arg.startswith("--mem") for arg in ls6_cmd)
    assert not any(arg.startswith("--gres") for arg in ls6_cmd)

    gattaca = slurm_profile("gattaca2")
    gattaca_cmd = build_sbatch_command(
        gattaca,
        script=Path("work/experiments/ml/hpc/sites/gattaca2/train_ml.sbatch"),
        job_name="S01-E01-R001",
        extra_args=("--partition=gpu", "--gres=gpu:1"),
    )
    assert "--account=shera_hpc" in gattaca_cmd
    assert "--mem=64G" in gattaca_cmd
    assert "--gres=gpu:1" in gattaca_cmd


def test_submit_helper_dry_run_uses_cli_identity_not_stale_shell(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ML_STUDY_PATH", "/stale/study.yaml")
    monkeypatch.setenv("ML_EXPERIMENT_ID", "S05-E01")
    monkeypatch.setenv("ML_RUN_ID", "S05-E01-R999")
    module = _submit_module()
    rc = module.main(
        [
            "--site",
            "tacc_ls6",
            "--study",
            str(S05_STUDY),
            "--experiment-id",
            "S05-E02",
            "--run-id",
            "S05-E02-R001",
            "--persist-artifact-root",
            str(tmp_path / "artifacts"),
            "--source-commit",
            "5d397eca9e206180785ce4b0d1593e19878c79b7",
            "--source-archive-id",
            "archive-sha256-test",
            "--dry-run",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["site"] == "tacc_ls6"
    assert payload["study_path"] == str(S05_STUDY.resolve())
    assert payload["experiment_id"] == "S05-E02"
    assert payload["run_id"] == "S05-E02-R001"
    assert payload["environment"]["ML_STUDY_PATH"] == str(S05_STUDY.resolve())
    assert payload["environment"]["ML_EXPERIMENT_ID"] == "S05-E02"
    assert payload["environment"]["ML_RUN_ID"] == "S05-E02-R001"
    assert payload["environment"]["ML_PERSIST_ARTIFACT_ROOT"] == str(
        (tmp_path / "artifacts").resolve()
    )
    assert (
        payload["environment"]["DLUXSHERA_SOURCE_COMMIT"]
        == "5d397eca9e206180785ce4b0d1593e19878c79b7"
    )
    assert payload["environment"]["DLUXSHERA_SOURCE_ARCHIVE_ID"] == "archive-sha256-test"
    assert "--export=ALL" in payload["command"]
    assert payload["missing_required_environment"] == [
        "ML_PREPARED_ROOT",
        "ML_SPLIT_REGISTRY",
        "ML_VALIDATION_MANIFEST",
        "ML_TEST_MANIFEST",
        "ML_RUN_DIR",
    ]


def test_submit_helper_creates_log_directory_before_sbatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    module = _submit_module()
    log_root = tmp_path / "slurm_logs"
    captured: dict[str, Any] = {}

    def fake_run(cmd, **kwargs):
        assert log_root.exists()
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(cmd, 0, stdout="576430;edge\n", stderr="")

    monkeypatch.setenv("ML_EXPERIMENT_ID", "stale")
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    rc = module.main(
        [
            "--site",
            "gattaca2",
            "--study",
            str(S05_STUDY),
            "--experiment-id",
            "S05-E02",
            "--run-id",
            "S05-E02-R001",
            "--prepared-root",
            str(tmp_path / "prepared"),
            "--split-registry",
            str(tmp_path / "split" / "SPLIT-ML-v1.json"),
            "--validation-manifest",
            str(tmp_path / "validation_pairs" / "S01-VALIDATION-PAIRS-v1"),
            "--test-manifest",
            str(tmp_path / "test_pairs" / "S01-TEST-PAIRS-v1"),
            "--run-dir",
            str(tmp_path / "runs" / "S05-E02-R001"),
            "--log-root",
            str(log_root),
            "--extra-sbatch-arg=-M",
            "--extra-sbatch-arg=edge",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["job_id"] == "576430"
    assert f"--output={log_root.resolve()}/%x-%j.out" in captured["cmd"]
    assert f"--error={log_root.resolve()}/%x-%j.err" in captured["cmd"]
    assert captured["env"]["ML_EXPERIMENT_ID"] == "S05-E02"
    assert captured["env"]["ML_RUN_ID"] == "S05-E02-R001"
    assert captured["env"]["ML_STUDY_PATH"] == str(S05_STUDY.resolve())


def test_submit_helper_resolves_repo_relative_paths_from_repo_root(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    module = _submit_module()
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    unrelated_cwd = tmp_path / "not_repo"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    rc = module.main(
        [
            "--site",
            "tacc_ls6",
            "--repo-root",
            str(fake_repo),
            "--study",
            "work/experiments/ml/s05/study.yaml",
            "--experiment-id",
            "S05-E01",
            "--run-id",
            "S05-E01-R001",
            "--dry-run",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["study_path"] == str(
        fake_repo / "work/experiments/ml/s05/study.yaml"
    )
    assert payload["environment"]["ML_REPO_ROOT"] == str(fake_repo.resolve())
    assert payload["environment"]["ML_STUDY_PATH"] == str(
        fake_repo / "work/experiments/ml/s05/study.yaml"
    )
    assert payload["command"][-1] == str(
        fake_repo / "work/experiments/ml/hpc/sites/tacc_ls6/train_ml.sbatch"
    )
    assert payload["output"] == str(
        fake_repo / "work/experiments/ml/hpc/logs/%x-%j.out"
    )
    assert payload["error"] == str(
        fake_repo / "work/experiments/ml/hpc/logs/%x-%j.err"
    )
    assert unrelated_cwd.as_posix() not in payload["study_path"]
    assert unrelated_cwd.as_posix() not in payload["command"][-1]


def test_s01_compatibility_submit_ignores_stale_generic_ml_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_path = tmp_path / "captured_env.json"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_sbatch = bin_dir / "sbatch"
    fake_sbatch.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json, os, pathlib",
                f"path = pathlib.Path({str(env_path)!r})",
                "keys = ['ML_STUDY_PATH', 'ML_EXPERIMENT_ID', 'ML_RUN_ID', "
                "'ML_REPO_ROOT', 'ML_PREPARED_ROOT', 'ML_SPLIT_REGISTRY', "
                "'ML_VALIDATION_MANIFEST', 'ML_TEST_MANIFEST', 'ML_RUN_DIR', "
                "'ML_PERSIST_DIR', 'ML_PERSIST_ARTIFACT_ROOT', "
                "'DLUXSHERA_SOURCE_COMMIT']",
                "path.write_text(json.dumps({key: os.environ.get(key) for key in keys}, "
                "sort_keys=True), encoding='utf-8')",
                "print('576430;edge')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USER", "testuser")
    monkeypatch.setenv("S01_REPO_ROOT", str(ROOT))
    monkeypatch.setenv("S01_SCRATCH_SIDE", "edge")
    monkeypatch.setenv("S01_GPU_SBATCH_ARGS", "--partition=gpu --gres=gpu:1")
    monkeypatch.setenv("S01_RUN_ID", "S01-E01-R777")
    monkeypatch.setenv("S01_SOURCE_COMMIT", "s01-source-commit")
    monkeypatch.setenv("ML_STUDY_PATH", "/stale/s05/study.yaml")
    monkeypatch.setenv("ML_EXPERIMENT_ID", "S05-E02")
    monkeypatch.setenv("ML_RUN_ID", "S05-E02-R001")
    monkeypatch.setenv("ML_PREPARED_ROOT", "/stale/prepared")
    monkeypatch.setenv("DLUXSHERA_SOURCE_COMMIT", "stale-source-commit")

    result = subprocess.run(
        [str(ROOT / "work/experiments/ml/s01/hpc/submit_s01_e01.sh")],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert "Parsed Slurm job ID: 576430" in result.stdout
    captured = json.loads(env_path.read_text(encoding="utf-8"))
    assert captured["ML_STUDY_PATH"] == "work/experiments/ml/s01/study.yaml"
    assert captured["ML_EXPERIMENT_ID"] == "S01-E01"
    assert captured["ML_RUN_ID"] == "S01-E01-R777"
    assert captured["ML_REPO_ROOT"] == str(ROOT)
    assert captured["ML_PREPARED_ROOT"].endswith("/data/PREP-V3-nuisance-v1")
    assert captured["ML_SPLIT_REGISTRY"].endswith("/artifacts/S01/split/SPLIT-ML-v1.json")
    assert captured["ML_VALIDATION_MANIFEST"].endswith(
        "/artifacts/S01/validation_pairs/S01-VALIDATION-PAIRS-v1"
    )
    assert captured["ML_TEST_MANIFEST"].endswith(
        "/artifacts/S01/test_pairs/S01-TEST-PAIRS-v1"
    )
    assert captured["ML_RUN_DIR"].endswith("/runs/S01/S01-E01/S01-E01-R777")
    assert captured["ML_PERSIST_DIR"].endswith("/S01/S01-E01/S01-E01-R777")
    assert captured["ML_PERSIST_ARTIFACT_ROOT"].endswith("/S01/artifacts")
    assert captured["DLUXSHERA_SOURCE_COMMIT"] == "s01-source-commit"


def test_prepare_sbatch_submission_reports_resolved_log_paths(tmp_path: Path) -> None:
    submission = prepare_sbatch_submission(
        slurm_profile("tacc_ls6"),
        script=Path("work/experiments/ml/hpc/sites/tacc_ls6/train_ml.sbatch"),
        job_name="S05-E01-R001",
        submitted_env={
            "ML_STUDY_PATH": S05_STUDY.resolve(),
            "ML_EXPERIMENT_ID": "S05-E01",
            "ML_RUN_ID": "S05-E01-R001",
            "ML_PREPARED_ROOT": tmp_path / "prepared",
            "ML_SPLIT_REGISTRY": tmp_path / "split.json",
            "ML_VALIDATION_MANIFEST": tmp_path / "validation",
            "ML_TEST_MANIFEST": tmp_path / "test",
            "ML_RUN_DIR": tmp_path / "run",
        },
        log_root=tmp_path / "logs",
        base_env={"ML_EXPERIMENT_ID": "stale", "PATH": "/bin"},
    )
    assert submission.missing_required_environment == ()
    assert submission.output == tmp_path.resolve() / "logs" / "%x-%j.out"
    assert submission.error == tmp_path.resolve() / "logs" / "%x-%j.err"
    assert submission.log_directories == (tmp_path.resolve() / "logs",)
    assert submission.environment["ML_EXPERIMENT_ID"] == "S05-E01"
    assert submission.environment["PATH"] == "/bin"


def test_parse_sbatch_job_id_accepts_real_site_outputs() -> None:
    assert parse_sbatch_job_id("3418708\n") == "3418708"
    assert parse_sbatch_job_id("576430;edge\n") == "576430"
    stdout = """-----------------------------------------------------------------
              Welcome to the Lonestar6 Supercomputer
-----------------------------------------------------------------
Validating project allocation
3418708
"""
    assert parse_sbatch_job_id(stdout) == "3418708"
    with pytest.raises(ValueError, match="Slurm job ID"):
        parse_sbatch_job_id("Welcome only\\n")
    with pytest.raises(ValueError, match="Slurm job ID"):
        parse_sbatch_job_id("576430;edge;extra\\n")


def test_persist_study_contract_artifacts_is_idempotent_and_compact(tmp_path: Path) -> None:
    catalog = load_sample_catalog(_write_prepared_fixture(tmp_path / "prepared"))
    registry = generate_split_registry(
        catalog,
        seed=7,
        science_fractions={"train": 0.34, "validation": 0.33, "test": 0.33},
        nuisance_fractions={"train": 0.34, "validation": 0.33, "test": 0.33},
    )
    study = _study(
        catalog.prepared_dataset_hash,
        split_content_sha256=split_registry_content_sha256(registry),
    )
    study["evaluation_artifacts"]["test"] = {
        **study["evaluation_artifacts"]["validation"],
        "artifact_id": "S01-TEST-PAIRS-v1",
        "split": "test",
        "seed": 1102,
        "eval_slices": {
            "heldout_science_seen_nuisance": {
                "science_split": "test",
                "nuisance_split": "train",
            },
            "heldout_science_heldout_nuisance": {
                "science_split": "test",
                "nuisance_split": "test",
            },
        },
    }
    policy = PairPolicy.from_dict(
        {
            "policy_id": "s01_clean_same_pair_grid_v1",
            **study["pair_policies"]["s01_clean_same_pair_grid_v1"],
        }
    )
    split_path = tmp_path / "source" / "split" / "SPLIT-ML-v1.json"
    write_split_registry(split_path, registry)
    manifests = {}
    for key, recipe in study["evaluation_artifacts"].items():
        manifest = generate_frozen_pair_manifest(
            catalog,
            registry,
            policy=policy,
            artifact_id=recipe["artifact_id"],
            split=recipe["split"],
            seed=recipe["seed"],
            pairs_per_slice=recipe["pairs_per_slice"],
            eval_slices=recipe["eval_slices"],
        )
        outdir = tmp_path / "source" / f"{key}_pairs" / recipe["artifact_id"]
        write_pair_manifest(outdir, manifest)
        manifests[key] = outdir

    artifact_root = tmp_path / "persisted" / "S01" / "artifacts"
    first = persist_study_contract_artifacts(
        artifact_root=artifact_root,
        split_registry_path=split_path,
        validation_manifest_path=manifests["validation"],
        test_manifest_path=manifests["test"],
    )
    second = persist_study_contract_artifacts(
        artifact_root=artifact_root,
        split_registry_path=split_path,
        validation_manifest_path=manifests["validation"],
        test_manifest_path=manifests["test"],
    )
    assert Path(first["split"]["destination"]) == artifact_root.resolve() / "split" / "SPLIT-ML-v1.json"
    assert Path(first["validation_pairs"]["destination"]) == (
        artifact_root.resolve() / "validation_pairs" / "S01-VALIDATION-PAIRS-v1"
    )
    assert Path(first["test_pairs"]["destination"]) == (
        artifact_root.resolve() / "test_pairs" / "S01-TEST-PAIRS-v1"
    )
    assert first["validation_pairs"]["status"] == "copied"
    assert second["validation_pairs"]["status"] == "exists"
    assert not (artifact_root / "data").exists()

    conflicting_split = artifact_root / "split" / "SPLIT-ML-v1.json"
    payload = json.loads(conflicting_split.read_text(encoding="utf-8"))
    payload["artifact_id"] = "DIFFERENT-SPLIT"
    write_json(conflicting_split, payload)
    with pytest.raises(FileExistsError, match="different artifact"):
        persist_study_contract_artifacts(
            artifact_root=artifact_root,
            split_registry_path=split_path,
            validation_manifest_path=manifests["validation"],
            test_manifest_path=manifests["test"],
        )


def test_persist_file_artifact_publication_race_accepts_same_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog = load_sample_catalog(_write_prepared_fixture(tmp_path / "prepared"))
    registry = generate_split_registry(
        catalog,
        seed=7,
        science_fractions={"train": 0.34, "validation": 0.33, "test": 0.33},
        nuisance_fractions={"train": 0.34, "validation": 0.33, "test": 0.33},
    )
    split_path = tmp_path / "source" / "split" / "SPLIT-ML-v1.json"
    write_split_registry(split_path, registry)
    destination = tmp_path / "persisted" / "split" / "SPLIT-ML-v1.json"
    original_link = hpc_module.os.link
    raced = False

    def racing_link(src, dst, *args, **kwargs):
        nonlocal raced
        if Path(dst) == destination and not raced:
            raced = True
            destination.parent.mkdir(parents=True, exist_ok=True)
            original_link(src, dst, *args, **kwargs)
            raise FileExistsError(str(dst))
        return original_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(hpc_module.os, "link", racing_link)
    result = hpc_module._copy_file_artifact(
        source=split_path,
        destination=destination,
        identity_fn=hpc_module._split_identity,
    )

    assert raced is True
    assert result["status"] == "exists"
    assert hpc_module._split_identity(destination) == hpc_module._split_identity(split_path)
    assert not list(destination.parent.glob("*.tmp-*"))


def test_persist_directory_artifact_publication_race_accepts_same_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog = load_sample_catalog(_write_prepared_fixture(tmp_path / "prepared"))
    registry = generate_split_registry(
        catalog,
        seed=7,
        science_fractions={"train": 0.34, "validation": 0.33, "test": 0.33},
        nuisance_fractions={"train": 0.34, "validation": 0.33, "test": 0.33},
    )
    study = _study(
        catalog.prepared_dataset_hash,
        split_content_sha256=split_registry_content_sha256(registry),
    )
    recipe = study["evaluation_artifacts"]["validation"]
    policy = PairPolicy.from_dict(
        {
            "policy_id": "s01_clean_same_pair_grid_v1",
            **study["pair_policies"]["s01_clean_same_pair_grid_v1"],
        }
    )
    manifest = generate_frozen_pair_manifest(
        catalog,
        registry,
        policy=policy,
        artifact_id=recipe["artifact_id"],
        split=recipe["split"],
        seed=recipe["seed"],
        pairs_per_slice=recipe["pairs_per_slice"],
        eval_slices=recipe["eval_slices"],
    )
    source = tmp_path / "source" / "validation_pairs" / recipe["artifact_id"]
    write_pair_manifest(source, manifest)
    destination = tmp_path / "persisted" / "validation_pairs" / recipe["artifact_id"]
    original_rename = Path.rename
    raced = False

    def racing_rename(self: Path, target: Path) -> Path:
        nonlocal raced
        if target == destination and not raced:
            raced = True
            shutil.copytree(source, destination)
            raise FileExistsError(str(target))
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", racing_rename)
    result = hpc_module._copy_directory_artifact(
        source=source,
        destination=destination,
        identity_fn=hpc_module._pair_manifest_identity,
    )

    assert raced is True
    assert result["status"] == "exists"
    assert hpc_module._pair_manifest_identity(destination) == hpc_module._pair_manifest_identity(source)
    assert not list(destination.parent.glob("*.tmp-*"))


def test_persist_directory_artifact_enotempty_race_accepts_same_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog = load_sample_catalog(_write_prepared_fixture(tmp_path / "prepared"))
    registry = generate_split_registry(
        catalog,
        seed=7,
        science_fractions={"train": 0.34, "validation": 0.33, "test": 0.33},
        nuisance_fractions={"train": 0.34, "validation": 0.33, "test": 0.33},
    )
    study = _study(
        catalog.prepared_dataset_hash,
        split_content_sha256=split_registry_content_sha256(registry),
    )
    recipe = study["evaluation_artifacts"]["validation"]
    policy = PairPolicy.from_dict(
        {
            "policy_id": "s01_clean_same_pair_grid_v1",
            **study["pair_policies"]["s01_clean_same_pair_grid_v1"],
        }
    )
    manifest = generate_frozen_pair_manifest(
        catalog,
        registry,
        policy=policy,
        artifact_id=recipe["artifact_id"],
        split=recipe["split"],
        seed=recipe["seed"],
        pairs_per_slice=recipe["pairs_per_slice"],
        eval_slices=recipe["eval_slices"],
    )
    source = tmp_path / "source" / "validation_pairs" / recipe["artifact_id"]
    write_pair_manifest(source, manifest)
    destination = tmp_path / "persisted" / "validation_pairs" / recipe["artifact_id"]
    original_rename = Path.rename
    raced = False

    def racing_rename(self: Path, target: Path) -> Path:
        nonlocal raced
        if target == destination and not raced:
            raced = True
            shutil.copytree(source, destination)
            raise OSError(errno.ENOTEMPTY, "Directory not empty", str(target))
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", racing_rename)
    result = hpc_module._copy_directory_artifact(
        source=source,
        destination=destination,
        identity_fn=hpc_module._pair_manifest_identity,
    )

    assert raced is True
    assert result["status"] == "exists"
    assert hpc_module._pair_manifest_identity(destination) == hpc_module._pair_manifest_identity(source)
    assert not list(destination.parent.glob("*.tmp-*"))


def test_persist_directory_artifact_enotempty_race_rejects_conflicting_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog = load_sample_catalog(_write_prepared_fixture(tmp_path / "prepared"))
    registry = generate_split_registry(
        catalog,
        seed=7,
        science_fractions={"train": 0.34, "validation": 0.33, "test": 0.33},
        nuisance_fractions={"train": 0.34, "validation": 0.33, "test": 0.33},
    )
    study = _study(
        catalog.prepared_dataset_hash,
        split_content_sha256=split_registry_content_sha256(registry),
    )
    recipe = study["evaluation_artifacts"]["validation"]
    policy = PairPolicy.from_dict(
        {
            "policy_id": "s01_clean_same_pair_grid_v1",
            **study["pair_policies"]["s01_clean_same_pair_grid_v1"],
        }
    )
    source_manifest = generate_frozen_pair_manifest(
        catalog,
        registry,
        policy=policy,
        artifact_id=recipe["artifact_id"],
        split=recipe["split"],
        seed=recipe["seed"],
        pairs_per_slice=recipe["pairs_per_slice"],
        eval_slices=recipe["eval_slices"],
    )
    conflicting_manifest = generate_frozen_pair_manifest(
        catalog,
        registry,
        policy=policy,
        artifact_id=recipe["artifact_id"],
        split=recipe["split"],
        seed=999,
        pairs_per_slice=recipe["pairs_per_slice"],
        eval_slices=recipe["eval_slices"],
    )
    source = tmp_path / "source" / "validation_pairs" / recipe["artifact_id"]
    destination = tmp_path / "persisted" / "validation_pairs" / recipe["artifact_id"]
    write_pair_manifest(source, source_manifest)
    original_rename = Path.rename
    raced = False

    def racing_rename(self: Path, target: Path) -> Path:
        nonlocal raced
        if target == destination and not raced:
            raced = True
            write_pair_manifest(destination, conflicting_manifest)
            raise OSError(errno.ENOTEMPTY, "Directory not empty", str(target))
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", racing_rename)
    with pytest.raises(FileExistsError, match="different artifact"):
        hpc_module._copy_directory_artifact(
            source=source,
            destination=destination,
            identity_fn=hpc_module._pair_manifest_identity,
        )

    assert raced is True
    assert hpc_module._pair_manifest_identity(destination) != hpc_module._pair_manifest_identity(source)
    assert not list(destination.parent.glob("*.tmp-*"))


def test_training_git_info_preserves_explicit_provenance_without_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    import dluxshera.ml.training as training

    monkeypatch.setenv("DLUXSHERA_SOURCE_COMMIT", "source-commit-test")
    monkeypatch.setenv("DLUXSHERA_SOURCE_ARCHIVE_ID", "source-archive-test")

    def missing_git(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(training.subprocess, "run", missing_git)
    info = training._git_info()
    assert info["source_commit"] == "source-commit-test"
    assert info["source_archive_id"] == "source-archive-test"
    assert info["commit"] is None
    assert info["branch"] is None
    assert info["dirty"] is None
    assert info["has_git_metadata"] is False


def test_hpc_shell_scripts_pass_bash_syntax_check() -> None:
    scripts = [
        ROOT / "work/experiments/ml/hpc/run_study_training.sh",
        ROOT / "work/experiments/ml/hpc/sites/gattaca2/train_ml.sbatch",
        ROOT / "work/experiments/ml/hpc/sites/tacc_ls6/train_ml.sbatch",
        ROOT / "work/experiments/ml/s01/hpc/submit_s01_e01.sh",
        ROOT / "work/experiments/ml/s01/hpc/train_s01_e01.sbatch",
    ]
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_generic_runner_conda_initialization_is_explicit_and_actionable() -> None:
    text = (ROOT / "work/experiments/ml/hpc/run_study_training.sh").read_text(
        encoding="utf-8"
    )
    assert 'source "$ML_CONDA_SH"' in text
    assert "command -v conda" in text
    assert "Conda is unavailable after initialization" in text
    assert 'conda activate "$ML_CONDA_ENV"' in text
    assert 'conda activate "$ML_CONDA_PREFIX"' in text
    assert "Set ML_CONDA_ENV or ML_CONDA_PREFIX" in text
    assert "/cm/shared/apps/miniforge" not in text


def test_site_wrappers_resolve_generic_runner_from_ml_repo_root() -> None:
    for script in (
        ROOT / "work/experiments/ml/hpc/sites/gattaca2/train_ml.sbatch",
        ROOT / "work/experiments/ml/hpc/sites/tacc_ls6/train_ml.sbatch",
    ):
        text = script.read_text(encoding="utf-8")
        assert 'REPO_ROOT="${ML_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"' in text
        assert 'exec bash "$REPO_ROOT/work/experiments/ml/hpc/run_study_training.sh"' in text
        assert "exec bash work/experiments/ml/hpc/run_study_training.sh" not in text


def test_s01_batch_wrapper_does_not_default_from_generic_ml_identity() -> None:
    text = (ROOT / "work/experiments/ml/s01/hpc/train_s01_e01.sbatch").read_text(
        encoding="utf-8"
    )
    assert 'export ML_STUDY_PATH="work/experiments/ml/s01/study.yaml"' in text
    assert 'export ML_EXPERIMENT_ID="S01-E01"' in text
    assert 'export ML_RUN_ID="$S01_RUN_ID_RESOLVED"' in text
    assert "${ML_STUDY_PATH:-" not in text
    assert "${ML_EXPERIMENT_ID:-" not in text
    assert "${ML_RUN_ID:-" not in text


def test_s01_conda_fallback_is_compatibility_only() -> None:
    generic = (ROOT / "work/experiments/ml/hpc/run_study_training.sh").read_text(
        encoding="utf-8"
    )
    assert "/cm/shared/apps/miniforge/etc/profile.d/conda.sh" not in generic

    for script in (
        ROOT / "work/experiments/ml/s01/hpc/submit_s01_e01.sh",
        ROOT / "work/experiments/ml/s01/hpc/train_s01_e01.sbatch",
    ):
        text = script.read_text(encoding="utf-8")
        explicit = text.index('S01_CONDA_SH_RESOLVED="${S01_CONDA_SH:-}"')
        fallback = text.index("/cm/shared/apps/miniforge/etc/profile.d/conda.sh")
        export = text.index('export ML_CONDA_SH="$S01_CONDA_SH_RESOLVED"')
        assert explicit < fallback < export
