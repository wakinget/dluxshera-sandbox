from __future__ import annotations

from pathlib import Path

import pytest

from dluxshera.ml import (
    PairPolicy,
    generate_frozen_pair_manifest,
    generate_split_registry,
    load_study_contract_artifacts,
    load_sample_catalog,
    resolve_study_experiment_config,
    split_registry_content_sha256,
    validate_evaluation_artifact_against_recipe,
    validate_prepared_dataset_for_study,
    validate_split_registry_for_study,
    write_pair_manifest,
    write_split_registry,
)
from dluxshera.datasets.schema import write_json
from tests.ml.test_catalog_splits_pairs import _write_prepared_fixture


def _study(
    catalog_hash: str = "hash",
    *,
    dataset_artifact_id: str = "PREP-V3-v1",
    split_content_sha256: str | None = None,
) -> dict:
    return {
        "study_id": "S01",
        "dataset": {"artifact_id": dataset_artifact_id, "prepared_dataset_hash": catalog_hash},
        "split_registry": {
            "artifact_id": "SPLIT-ML-v1",
            **({} if split_content_sha256 is None else {"content_sha256": split_content_sha256}),
        },
        "pair_policies": {
            "s01_clean_same_pair_grid_v1": {
                "family_weights": {"same_nuisance_different_science": 1.0},
                "same_pair_id": True,
                "min_fisher_distance": 0.0,
                "max_fisher_distance": 5000.0,
                "include_reverse": True,
                "max_sampling_attempts": 4000,
            }
        },
        "evaluation_artifacts": {
            "validation": {
                "artifact_id": "S01-VALIDATION-PAIRS-v1",
                "pair_policy_id": "s01_clean_same_pair_grid_v1",
                "split": "validation",
                "seed": 1101,
                "pairs_per_slice": 2,
                "eval_slices": {
                    "heldout_science_seen_nuisance": {
                        "science_split": "validation",
                        "nuisance_split": "train",
                    },
                    "heldout_science_heldout_nuisance": {
                        "science_split": "validation",
                        "nuisance_split": "validation",
                    },
                },
            }
        },
        "experiments": {
            "S01-E01": {
                "run_id": "S01-E01-R001",
                "seed": 11,
                "device": "cuda:0",
                "pair_policy_id": "s01_clean_same_pair_grid_v1",
                "evaluate_test": False,
                "training": {"epochs": 100},
            }
        },
    }


def test_s01_e01_resolves_study_level_policy_by_id() -> None:
    config = resolve_study_experiment_config(_study(), experiment_id="S01-E01")
    assert config["study_id"] == "S01"
    assert config["experiment_id"] == "S01-E01"
    assert config["run_id"] == "S01-E01-R001"
    assert config["pair_policy"]["policy_id"] == "s01_clean_same_pair_grid_v1"
    assert config["pair_policy"]["same_pair_id"] is True
    assert config["pair_policy"]["min_fisher_distance"] == 0.0
    assert config["pair_policy"]["max_fisher_distance"] == 5000.0
    assert config["pair_policy"]["include_reverse"] is True
    assert config["pair_policy"]["max_sampling_attempts"] == 4000


def test_unknown_policy_id_fails_clearly() -> None:
    study = _study()
    study["experiments"]["S01-E01"]["pair_policy_id"] = "missing"
    with pytest.raises(ValueError, match="Unknown pair_policy_id"):
        resolve_study_experiment_config(study, experiment_id="S01-E01")


def test_invalid_policy_definition_fails_clearly() -> None:
    study = _study()
    study["pair_policies"]["s01_clean_same_pair_grid_v1"]["family_weights"] = {"bad": 1.0}
    with pytest.raises(ValueError, match="Unsupported pair family"):
        resolve_study_experiment_config(study, experiment_id="S01-E01")


def test_prepared_dataset_hash_guard(tmp_path: Path) -> None:
    catalog = load_sample_catalog(_write_prepared_fixture(tmp_path / "prepared"))
    validate_prepared_dataset_for_study(catalog, _study(catalog.prepared_dataset_hash))
    with pytest.raises(ValueError, match="Prepared dataset hash"):
        validate_prepared_dataset_for_study(catalog, _study("wrong"))


def test_prepared_dataset_artifact_id_guard_and_no_study_relabel(tmp_path: Path) -> None:
    prepared = _write_prepared_fixture(tmp_path / "prepared")
    catalog = load_sample_catalog(prepared)
    assert catalog.artifact_id == "PREP-V3-v1"
    with pytest.raises(ValueError, match="Prepared dataset artifact_id"):
        validate_prepared_dataset_for_study(
            catalog,
            _study(catalog.prepared_dataset_hash, dataset_artifact_id="PREP-V3-nuisance-v1"),
        )

    registry = generate_split_registry(
        catalog,
        seed=7,
        science_fractions={"train": 0.34, "validation": 0.33, "test": 0.33},
        nuisance_fractions={"train": 0.34, "validation": 0.33, "test": 0.33},
    )
    split_path = tmp_path / "split.json"
    write_split_registry(split_path, registry)
    with pytest.raises(ValueError, match="Prepared dataset artifact_id"):
        load_study_contract_artifacts(
            study=_study(
                catalog.prepared_dataset_hash,
                dataset_artifact_id="PREP-V3-nuisance-v1",
                split_content_sha256=split_registry_content_sha256(registry),
            ),
            prepared_root=prepared,
            split_registry_path=split_path,
        )


def test_split_registry_content_guard_rejects_same_name_wrong_assignments(tmp_path: Path) -> None:
    catalog = load_sample_catalog(_write_prepared_fixture(tmp_path / "prepared"))
    registry = generate_split_registry(
        catalog,
        seed=7,
        science_fractions={"train": 0.34, "validation": 0.33, "test": 0.33},
        nuisance_fractions={"train": 0.34, "validation": 0.33, "test": 0.33},
    )
    validate_split_registry_for_study(
        registry,
        _study(
            catalog.prepared_dataset_hash,
            split_content_sha256=split_registry_content_sha256(registry),
        ),
    )
    with pytest.raises(ValueError, match="Split registry content_sha256"):
        validate_split_registry_for_study(
            registry,
            _study(catalog.prepared_dataset_hash, split_content_sha256="wrong"),
        )

    payload = registry.to_dict()
    first_group = next(iter(payload["science_assignments"]))
    payload["science_assignments"][first_group] = (
        "validation"
        if payload["science_assignments"][first_group] != "validation"
        else "test"
    )
    changed_path = tmp_path / "changed_split.json"
    write_json(changed_path, payload)
    with pytest.raises(ValueError, match="Split registry content_sha256"):
        load_study_contract_artifacts(
            study=_study(
                catalog.prepared_dataset_hash,
                split_content_sha256=split_registry_content_sha256(registry),
            ),
            prepared_root=catalog.root,
            split_registry_path=changed_path,
        )


def test_frozen_artifact_recipe_validation_and_identity_stability(tmp_path: Path) -> None:
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
    first = generate_frozen_pair_manifest(
        catalog,
        registry,
        policy=policy,
        artifact_id=recipe["artifact_id"],
        split=recipe["split"],
        seed=recipe["seed"],
        pairs_per_slice=recipe["pairs_per_slice"],
        eval_slices=recipe["eval_slices"],
    )
    second = generate_frozen_pair_manifest(
        catalog,
        registry,
        policy=policy,
        artifact_id=recipe["artifact_id"],
        split=recipe["split"],
        seed=recipe["seed"],
        pairs_per_slice=recipe["pairs_per_slice"],
        eval_slices=recipe["eval_slices"],
    )
    assert [record.to_dict() for record in first.records] == [
        record.to_dict() for record in second.records
    ]
    assert first.manifest["content_identity"]["sha256"] == second.manifest["content_identity"]["sha256"]
    assert all(first.records[index + 1].sample_a_id == first.records[index].sample_b_id for index in range(0, len(first.records), 2))
    validate_evaluation_artifact_against_recipe(
        first,
        study=study,
        artifact_key="validation",
        split_registry=registry,
    )


def test_study_contract_rejects_wrong_validation_manifest_before_training(tmp_path: Path) -> None:
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
    split_path = tmp_path / "split.json"
    write_split_registry(split_path, registry)
    policy = PairPolicy.from_dict(
        {
            "policy_id": "s01_clean_same_pair_grid_v1",
            **study["pair_policies"]["s01_clean_same_pair_grid_v1"],
        }
    )
    wrong = generate_frozen_pair_manifest(
        catalog,
        registry,
        policy=policy,
        artifact_id="S01-VALIDATION-PAIRS-v1",
        split="validation",
        seed=999,
        pairs_per_slice=2,
        eval_slices=study["evaluation_artifacts"]["validation"]["eval_slices"],
    )
    wrong_path = tmp_path / "wrong_validation"
    write_pair_manifest(wrong_path, wrong)
    config = resolve_study_experiment_config(study, experiment_id="S01-E01")
    config["require_frozen_validation_manifest"] = True
    with pytest.raises(ValueError, match="does not match study recipe"):
        load_study_contract_artifacts(
            study=study,
            prepared_root=catalog.root,
            split_registry_path=split_path,
            validation_manifest_path=wrong_path,
            experiment_id="S01-E01",
            config=config,
        )


def test_frozen_artifact_ordered_pair_count_is_contract_field(tmp_path: Path) -> None:
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
        recipe["ordered_pair_count"] = len(manifest.records)
        manifests[key] = manifest

    validate_evaluation_artifact_against_recipe(
        manifests["validation"],
        study=study,
        artifact_key="validation",
        split_registry=registry,
    )
    validate_evaluation_artifact_against_recipe(
        manifests["test"],
        study=study,
        artifact_key="test",
        split_registry=registry,
    )

    study["evaluation_artifacts"]["validation"]["ordered_pair_count"] += 1
    with pytest.raises(ValueError, match="ordered_pair_count"):
        validate_evaluation_artifact_against_recipe(
            manifests["validation"],
            study=study,
            artifact_key="validation",
            split_registry=registry,
        )

    del study["evaluation_artifacts"]["validation"]["ordered_pair_count"]
    validate_evaluation_artifact_against_recipe(
        manifests["validation"],
        study=study,
        artifact_key="validation",
        split_registry=registry,
    )
