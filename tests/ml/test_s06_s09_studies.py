from __future__ import annotations

from pathlib import Path

from dluxshera.ml import PairPolicy, expand_study_run_plan, load_study_prescription, resolve_study_experiment_config


ROOT = Path("work/experiments/ml")
REPO_ROOT = Path(__file__).resolve().parents[2]


def _rows(study_id: str):
    study = load_study_prescription(ROOT / study_id.lower() / "study.yaml")
    return study, expand_study_run_plan(study)


def test_s06_s09_expand_to_exact_40_run_matrix() -> None:
    expected = {"S06": 9, "S07": 10, "S08": 12, "S09": 9}
    total = 0
    for study_id, count in expected.items():
        _, rows = _rows(study_id)
        assert len(rows) == count
        total += len(rows)
    assert total == 40


def test_s06_s09_exact_seeds_run_ids_and_test_eval_disabled() -> None:
    expected = {
        "S06": {
            "S06-E01": [11, 23, 47],
            "S06-E02": [11, 23, 47],
            "S06-E03": [11, 23, 47],
        },
        "S07": {
            "S07-E01": [11, 23, 47],
            "S07-E02": [11, 23, 47],
            "S07-E03": [11],
            "S07-E04": [11],
            "S07-E05": [11],
            "S07-E06": [11],
        },
        "S08": {
            "S08-E01": [11, 23, 47],
            "S08-E02": [11, 23, 47],
            "S08-E03": [11, 23, 47],
            "S08-E04": [11, 23, 47],
        },
        "S09": {
            "S09-E01": [11, 23, 47],
            "S09-E02": [11, 23, 47],
            "S09-E03": [11, 23, 47],
        },
    }
    for study_id, experiments in expected.items():
        study, rows = _rows(study_id)
        by_experiment = {experiment_id: [] for experiment_id in experiments}
        for row in rows:
            by_experiment[row.experiment_id].append(row)
            config = resolve_study_experiment_config(
                study,
                experiment_id=row.experiment_id,
                run_id=row.run_id,
            )
            assert config["evaluate_test"] is False
        for experiment_id, seeds in experiments.items():
            exp_rows = by_experiment[experiment_id]
            assert [row.seed for row in exp_rows] == seeds
            assert [row.run_id for row in exp_rows] == [
                f"{experiment_id}-R{idx:03d}" for idx in range(1, len(seeds) + 1)
            ]


def _policy(study: dict, experiment_id: str) -> PairPolicy:
    config = resolve_study_experiment_config(study, experiment_id=experiment_id)
    return PairPolicy.from_dict(config["pair_policy"])


def test_s09_family_specific_distance_sampling_semantics() -> None:
    study, _ = _rows("S09")
    e01 = _policy(study, "S09-E01").resolved_dataset_family_constraints()
    assert e01["joint_full_v4"]["weight"] == 0.75
    assert e01["radial_capture_v4"]["weight"] == 0.25
    assert e01["joint_full_v4"]["distance_bin_weights"] == {}
    assert e01["joint_full_v4"]["max_fisher_distance"] is None
    assert e01["radial_capture_v4"]["distance_bin_weights"] == {
        "0-100": 1.0,
        "100-250": 1.0,
        "250-500": 1.0,
        "500-1000": 1.0,
        "1000-2000": 1.0,
        "2000-5000": 1.0,
    }

    e02 = _policy(study, "S09-E02").resolved_dataset_family_constraints()
    assert e02["joint_full_v4"]["weight"] == 0.5
    assert e02["radial_capture_v4"]["weight"] == 0.5
    assert e02["joint_full_v4"]["distance_bin_weights"] == {}
    assert e02["radial_capture_v4"]["distance_bin_weights"] == e01["radial_capture_v4"]["distance_bin_weights"]


def test_s09_curriculum_updates_radial_family_only() -> None:
    study, _ = _rows("S09")
    policy = _policy(study, "S09-E03")

    epoch0 = policy.resolved_dataset_family_constraints(epoch=0)
    assert epoch0["joint_full_v4"]["distance_bin_weights"] == {}
    assert epoch0["joint_full_v4"]["max_fisher_distance"] is None
    assert set(epoch0["radial_capture_v4"]["distance_bin_weights"]) == {
        "0-100",
        "100-250",
        "250-500",
    }

    epoch100 = policy.resolved_dataset_family_constraints(epoch=100)
    assert epoch100["joint_full_v4"] == epoch0["joint_full_v4"]
    assert set(epoch100["radial_capture_v4"]["distance_bin_weights"]) == {
        "0-100",
        "100-250",
        "250-500",
        "500-1000",
        "1000-2000",
    }

    epoch250 = policy.resolved_dataset_family_constraints(epoch=250)
    assert epoch250["joint_full_v4"] == epoch0["joint_full_v4"]
    assert set(epoch250["radial_capture_v4"]["distance_bin_weights"]) == {
        "0-100",
        "100-250",
        "250-500",
        "500-1000",
        "1000-2000",
        "2000-5000",
    }


def test_s09_primary_validation_recipe_preserves_family_specific_semantics() -> None:
    study, _ = _rows("S09")
    recipe = study["evaluation_artifacts"]["validation"]
    policy_id = recipe["pair_policy_id"]
    policy = PairPolicy.from_dict({"policy_id": policy_id, **study["pair_policies"][policy_id]})
    constraints = policy.resolved_dataset_family_constraints()
    assert constraints["joint_full_v4"]["weight"] == 0.5
    assert constraints["joint_full_v4"]["distance_bin_weights"] == {}
    assert constraints["radial_capture_v4"]["weight"] == 0.5
    assert len(constraints["radial_capture_v4"]["distance_bin_weights"]) == 6


def test_s08_artifact_profiles_resolve_standard_and_holdout_contracts() -> None:
    study, rows = _rows("S08")
    by_exp = {row.experiment_id: row for row in rows if row.run_id.endswith("R001")}
    assert by_exp["S08-E01"].artifact_profile == "standard"
    assert by_exp["S08-E01"].validation_artifact == "validation_c"
    assert by_exp["S08-E01"].split_registry_artifact_id == "SPLIT-V4-ROLE-PRESERVING-v1"
    assert by_exp["S08-E01"].artifact_lock_id == "S08-STANDARD-ARTIFACT-LOCK-v1"

    for experiment_id in ("S08-E02", "S08-E03"):
        assert by_exp[experiment_id].artifact_profile == "standard"
        assert by_exp[experiment_id].validation_artifact == "validation_abc"
        assert by_exp[experiment_id].split_registry_artifact_id == "SPLIT-V4-ROLE-PRESERVING-v1"

    assert by_exp["S08-E04"].artifact_profile == "nuisance_holdout"
    assert by_exp["S08-E04"].validation_artifact == "validation_abc_seen"
    assert by_exp["S08-E04"].test_artifact == "test_seen"
    assert by_exp["S08-E04"].split_registry_artifact_id == "SPLIT-V4-S08-NUISANCE-HOLDOUT-v1"
    assert by_exp["S08-E04"].artifact_lock_id == "S08-HOLDOUT-ARTIFACT-LOCK-v1"
    assert by_exp["S08-E04"].audit_artifacts == ("unseen_nuisance_audit",)

    e04 = resolve_study_experiment_config(study, experiment_id="S08-E04")
    assert e04["primary_validation_artifact_key"] == "validation_abc_seen"
    assert e04["test_artifact_key"] == "test_seen"
    assert e04["declared_audit_artifact_keys"] == ["unseen_nuisance_audit"]
    assert e04["split_registry_artifact_id"] == "SPLIT-V4-S08-NUISANCE-HOLDOUT-v1"
    assert e04["artifact_lock_id"] == "S08-HOLDOUT-ARTIFACT-LOCK-v1"
    assert e04["nuisance_holdout"]["train_nuisance_bank_indices"] == list(range(8))
    assert e04["nuisance_holdout"]["unseen_nuisance_bank_indices"] == [8, 9]


def test_campaign_plan_note_is_canonical_and_linked() -> None:
    note = REPO_ROOT / "docs" / "dev" / "notes" / "ml_s06_s09_campaign_plan.md"
    assert note.exists()
    working_plan = (REPO_ROOT / "docs" / "dev" / "working_plan.md").read_text(encoding="utf-8")
    assert "docs/dev/notes/ml_s06_s09_campaign_plan.md" in working_plan
    inverse = (REPO_ROOT / "docs" / "dev" / "shera_ml_inverse_model_design.md").read_text(encoding="utf-8")
    assert "| S06 | V3 Architecture / Training Closure |" in inverse
    for doc in ("working_plan.md", "shera_ml_inverse_model_design.md"):
        text = (REPO_ROOT / "docs" / "dev" / doc).read_text(encoding="utf-8")
        for line in text.splitlines():
            if "ml_s06_s09_campaign_plan.md" in line:
                assert "context" not in line.lower()
                assert "generated" not in line.lower()
