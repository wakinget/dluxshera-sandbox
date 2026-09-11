from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dluxshera.ml import (
    build_science_mode_weights,
    PairPolicy,
    build_science_eigenbasis,
    build_science_eigenbasis_from_source,
    build_s10_nominal_physical_fim_source,
    expand_study_run_plan,
    generate_frozen_pair_manifest,
    generate_split_registry,
    load_study_contract_artifacts,
    load_sample_catalog,
    load_study_prescription,
    resolve_study_experiment_config,
    science_eigenbasis_content_sha256,
    validate_noisy_eval_artifact_for_study,
    validate_science_eigenbasis_expectations,
    write_pair_manifest,
    write_split_registry,
)
import dluxshera.ml.eigenbasis as eigenbasis_module
from dluxshera.ml.studies import (
    validate_noisy_eval_artifact_for_study as studies_validate_noisy_eval_artifact_for_study,
)
from dluxshera.ml.eigenbasis import DEFAULT_COORDINATE_CONVENTION
from dluxshera.ml.noise import NoiseConfig, apply_pair_noise, noise_config_identity, pair_noise_side_seeds
from tests.ml.test_catalog_splits_pairs import _write_prepared_fixture

ROOT = Path("work/experiments/ml")


def test_noisy_eval_validator_is_exported_from_ml_package() -> None:
    assert validate_noisy_eval_artifact_for_study is studies_validate_noisy_eval_artifact_for_study


def test_s10_s12_audit_expands_to_21_runs_with_one_clean_reference_cohort() -> None:
    studies = [load_study_prescription(ROOT / name / "study.yaml") for name in ("s10", "s11", "s12")]
    rows = [row for study in studies for row in expand_study_run_plan(study)]
    assert len(rows) == 21
    assert sum(row.study_id == "S10" for row in rows) == 9
    assert sum(row.study_id == "S11" for row in rows) == 6
    assert sum(row.study_id == "S12" for row in rows) == 6
    assert sum(row.experiment_id == "S10-E01" for row in rows) == 3
    assert any("science_eigenbasis" in row.auxiliary_artifacts for row in rows if row.study_id == "S10")
    assert any("noisy_validation_recipe" in row.auxiliary_artifacts for row in rows if row.study_id == "S12")
    assert all(
        resolve_study_experiment_config(
            study,
            experiment_id=row.experiment_id,
            run_id=row.run_id,
        )["evaluate_test"]
        is False
        for study in studies
        for row in expand_study_run_plan(study)
    )
    assert studies[1]["shared_reference"]["run_ids"] == [
        "S10-E01-R001",
        "S10-E01-R002",
        "S10-E01-R003",
    ]
    assert studies[2]["shared_reference"]["retrain_in_this_study"] is False


def test_s10_study_weighted_variants_require_fixed_eigenbasis_identity() -> None:
    study = load_study_prescription(ROOT / "s10" / "study.yaml")
    ordinary = resolve_study_experiment_config(study, experiment_id="S10-E01")
    strong = resolve_study_experiment_config(study, experiment_id="S10-E02")
    weak = resolve_study_experiment_config(study, experiment_id="S10-E03")
    assert ordinary["science_loss"]["mode"] == "ordinary"
    assert strong["science_loss"]["mode"] == "strong_mode_weighted"
    assert weak["science_loss"]["mode"] == "weak_mode_weighted"
    assert strong["science_loss"]["eigenbasis"]["artifact_id"] == "S10-V4-SCIENCE-FIM-EIGENBASIS-v1"
    assert weak["science_loss"]["eigenvalue_floor"] > strong["science_loss"]["eigenvalue_floor"]
    assert weak["science_loss"]["weight_cap"] == 5.0


def test_s10_eigenbasis_expectation_validation_checks_coordinate_spaces(tmp_path: Path) -> None:
    catalog = load_sample_catalog(_write_prepared_fixture(tmp_path / "prepared"))
    basis = build_science_eigenbasis(
        artifact_id="S10-V4-SCIENCE-FIM-EIGENBASIS-v1",
        parameter_labels=catalog.parameter_labels,
        curvature_matrix=[[1.0, 0.25], [0.25, 1.0]],
        source_matrix_coordinate_space="physical_theta",
        fisher_scales=catalog.fisher_sigmas,
        source_provenance={"source_artifact_id": "s10_science_fim_source"},
    )
    validate_science_eigenbasis_expectations(
        basis,
        expected={
            "artifact_id": basis.artifact_id,
            "source_matrix_coordinate_space": "physical_theta",
            "eigenbasis_coordinate_space": "prepared_v4_fisher_scaled_science_delta",
            "source_fim_artifact_id": "s10_science_fim_source",
        },
    )
    with pytest.raises(ValueError, match="source_matrix_coordinate_space"):
        validate_science_eigenbasis_expectations(
            basis,
            expected={
                "artifact_id": basis.artifact_id,
                "source_matrix_coordinate_space": "fisher_scaled_z",
            },
        )


def test_s10_eigenbasis_hash_and_ordering_are_stable(tmp_path: Path) -> None:
    catalog = load_sample_catalog(_write_prepared_fixture(tmp_path / "prepared"))
    basis = build_science_eigenbasis(
        artifact_id="BASIS-v1",
        parameter_labels=catalog.parameter_labels,
        curvature_matrix=[[2.0, 0.5], [0.5, 4.0]],
        fisher_scales=catalog.fisher_sigmas,
    )
    same = build_science_eigenbasis(
        artifact_id="BASIS-v1",
        parameter_labels=catalog.parameter_labels,
        curvature_matrix=[[2.0, 0.5], [0.5, 4.0]],
        fisher_scales=catalog.fisher_sigmas,
    )
    assert basis.coordinate_convention == DEFAULT_COORDINATE_CONVENTION
    assert basis.parameter_labels == catalog.parameter_labels
    assert basis.content_identity["sha256"] == science_eigenbasis_content_sha256(same)
    assert np.all(np.diff(basis.eigenvalues) <= 0.0)
    with pytest.raises(ValueError, match="parameter_labels"):
        build_science_eigenbasis(
            artifact_id="WRONG-v1",
            parameter_labels=tuple(reversed(catalog.parameter_labels)),
            curvature_matrix=[[2.0, 0.0], [0.0, 4.0]],
            fisher_scales=catalog.fisher_sigmas,
        ).validate_catalog(catalog)


def test_s10_fim_source_coordinate_contracts_are_enforced(tmp_path: Path) -> None:
    catalog = load_sample_catalog(_write_prepared_fixture(tmp_path / "prepared"))
    physical = {
        "coordinate_space": "physical_theta",
        "parameter_labels": list(catalog.parameter_labels),
        "curvature_matrix": [[4.0, 0.25], [0.25, 0.25]],
    }
    basis = build_science_eigenbasis_from_source(
        artifact_id="BASIS-v1",
        source=physical,
        catalog=catalog,
    )
    np.testing.assert_allclose(basis.curvature_matrix, [[1.0, 0.25], [0.25, 1.0]])
    assert basis.physical_fim_identity["coordinate_space"] == "physical_theta"
    assert basis.transformed_fz_identity["coordinate_space"] == "prepared_v4_fisher_scaled_science_delta"

    missing_space = dict(physical)
    missing_space.pop("coordinate_space")
    with pytest.raises(ValueError, match="coordinate_space"):
        build_science_eigenbasis_from_source(
            artifact_id="BASIS-v1",
            source=missing_space,
            catalog=catalog,
        )
    with pytest.raises(ValueError, match="Unsupported science FIM coordinate_space"):
        build_science_eigenbasis_from_source(
            artifact_id="BASIS-v1",
            source={**physical, "coordinate_space": "raw_theta"},
            catalog=catalog,
        )
    with pytest.raises(ValueError, match="diagonal-derived Fisher sigmas"):
        build_science_eigenbasis_from_source(
            artifact_id="BASIS-v1",
            source={**physical, "curvature_matrix": [[1.0, 0.0], [0.0, 0.25]]},
            catalog=catalog,
        )
    with pytest.raises(ValueError, match="materially indefinite"):
        build_science_eigenbasis_from_source(
            artifact_id="BASIS-v1",
            source={**physical, "curvature_matrix": [[4.0, 5.0], [5.0, 0.25]]},
            catalog=catalog,
        )


def test_s10_nominal_fim_source_reproduces_prepared_fisher_scales(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = load_sample_catalog(_write_prepared_fixture(tmp_path / "prepared"))
    f_theta = np.asarray([[4.0, 0.25], [0.25, 0.25]], dtype=np.float64)

    def fake_compute(catalog):
        return {
            "fim_theta": f_theta,
            "parameter_labels": list(catalog.parameter_labels),
            "theta_ref": [0.0, 0.0],
            "index_map": {"entries": []},
            "image_shape": [8, 8],
            "system_cfg": {"preset": "unit-test"},
            "loss_convention": {
                "maker": "dluxshera.inference.optimization.make_binder_nll_fn",
                "fim": "dluxshera.inference.optimization.fim_theta",
                "var": "data",
                "noise_model": "gaussian",
                "reduce": "sum",
            },
            "source_implementation": "unit-test",
            "script_version": "unit-test",
            "git_info": {"commit": "unit-test"},
        }

    monkeypatch.setattr(
        eigenbasis_module,
        "_compute_s10_nominal_full_physical_fim",
        fake_compute,
    )
    source = build_s10_nominal_physical_fim_source(catalog=catalog)
    assert source["coordinate_space"] == "physical_theta"
    assert source["diagonal_fisher_scale_compatibility"]["status"] == "PASS"
    np.testing.assert_allclose(source["fisher_scales"], catalog.fisher_sigmas)
    assert source["curvature_matrix"][0][1] == pytest.approx(0.25)
    basis = build_science_eigenbasis_from_source(
        artifact_id="BASIS-v1",
        source=source,
        catalog=catalog,
    )
    assert not np.allclose(basis.curvature_matrix, np.eye(catalog.science_dim))
    assert basis.curvature_matrix[0, 1] == pytest.approx(0.25)
    for mode in ("strong_mode_weighted", "weak_mode_weighted"):
        weights = build_science_mode_weights(
            basis.eigenvalues,
            mode=mode,
            strength=0.5,
        )
        assert not np.allclose(weights, np.ones_like(weights))


def test_make_s10_fim_source_reexecs_with_process_start_x64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from work.experiments.ml import materialize_study_artifacts as materializer

    calls = []

    def fake_execvpe(file, args, env):
        calls.append((file, args, env))
        raise RuntimeError("reexec")

    monkeypatch.delenv("JAX_ENABLE_X64", raising=False)
    monkeypatch.setattr(materializer.os, "execvpe", fake_execvpe)
    monkeypatch.setattr(materializer.sys, "argv", ["materialize_study_artifacts.py", "make-s10-fim-source"])

    with pytest.raises(RuntimeError, match="reexec"):
        materializer._ensure_s10_fim_source_process_start_x64(["make-s10-fim-source"])

    assert calls
    assert calls[0][1] == [materializer.sys.executable, "materialize_study_artifacts.py", "make-s10-fim-source"]
    assert calls[0][2]["JAX_ENABLE_X64"] == "1"


def test_s12_noise_config_identity_records_physical_ordering() -> None:
    cfg = NoiseConfig(
        enabled=True,
        apply_to="observation",
        noise_model="shera_observation",
        photon_noise=True,
        read_noise=False,
        dark_current=False,
        seed=12030,
        training_dynamic=True,
        negative_policy="clip",
    )
    identity = noise_config_identity(cfg)
    assert identity["noise_model"] == "shera_observation"
    assert "before IntensityScaler.transform" in identity["physical_ordering"]
    image = np.full((4, 4), 100.0, dtype=np.float32)
    _, first = apply_pair_noise(image, image, cfg, pair_record_id="pair", dynamic_seed_offset=1)
    _, again = apply_pair_noise(image, image, cfg, pair_record_id="pair", dynamic_seed_offset=1)
    _, different = apply_pair_noise(image, image, cfg, pair_record_id="pair", dynamic_seed_offset=2)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, different)


def test_s12_study_uses_dynamic_training_noise_and_fixed_validation_noise() -> None:
    study = load_study_prescription(ROOT / "s12" / "study.yaml")
    e01 = resolve_study_experiment_config(study, experiment_id="S12-E01")
    e02 = resolve_study_experiment_config(study, experiment_id="S12-E02")
    assert e01["noise"]["training_dynamic"] is True
    assert e01["validation_noise"]["training_dynamic"] is False
    assert e01["validation_noise"]["seed"] == 12031
    assert e02["pair_consistency"]["noise_consistency_weight"] == 0.05
    assert e01["noise"]["photon_noise"] is True
    assert e01["noise"]["read_noise"] is False
    assert e01["noise"]["dark_current"] is False
    recipe = study["auxiliary_artifacts"]["noisy_validation_recipe"]
    assert recipe["artifact_id"] == "S12-PHOTON-NOISE-VALIDATION-v1"
    assert recipe["schema_version"] == "dluxshera_ml_noisy_eval_recipe/1"


def test_noisy_eval_materializer_records_per_pair_seed_identity(tmp_path: Path) -> None:
    prepared = _write_prepared_fixture(tmp_path / "prepared")
    catalog = load_sample_catalog(prepared)
    registry = generate_split_registry(
        catalog,
        seed=7,
        science_fractions={"train": 1.0, "validation": 0.0, "test": 0.0},
        nuisance_fractions={"train": 1.0, "validation": 0.0, "test": 0.0},
    )
    split_path = tmp_path / "split.json"
    write_split_registry(split_path, registry)
    policy = PairPolicy(
        policy_id="fixture_policy",
        family_weights={"A": 1.0},
        same_pair_id=True,
        min_fisher_distance=0.5,
        max_fisher_distance=3.0,
        include_reverse=False,
        max_sampling_attempts=4000,
    )
    manifest = generate_frozen_pair_manifest(
        catalog,
        registry,
        policy=policy,
        artifact_id="FIXTURE-VALIDATION-v1",
        split="validation",
        seed=4,
        pairs_per_slice=2,
        eval_slices={"fixture": {"science_split": "train", "nuisance_split": "train"}},
    )
    manifest_path = tmp_path / "pairs"
    write_pair_manifest(manifest_path, manifest)
    study_path = tmp_path / "study.yaml"
    study_path.write_text(
        """
study_id: TS12
dataset: {artifact_id: PREP-V3-v1}
split_registry: {artifact_id: SPLIT-ML-v1}
pair_policies:
  fixture_policy:
    family_weights: {A: 1.0}
    same_pair_id: true
    min_fisher_distance: 0.5
    max_fisher_distance: 3.0
    include_reverse: false
    max_sampling_attempts: 4000
evaluation_artifacts:
  validation:
    artifact_id: FIXTURE-VALIDATION-v1
    pair_policy_id: fixture_policy
    split: validation
    seed: 4
    pairs_per_slice: 2
    eval_slices: {fixture: {science_split: train, nuisance_split: train}}
defaults:
  pair_policy_id: fixture_policy
  validation_artifact: validation
  evaluate_test: false
  validation_noise:
    enabled: true
    apply_to: observation
    noise_model: shera_observation
    photon_noise: true
    read_noise: false
    dark_current: false
    seed: 12031
    training_dynamic: false
    negative_policy: clip
experiments:
  TS12-E01:
    seeds: [11]
""",
        encoding="utf-8",
    )
    from work.experiments.ml.materialize_study_artifacts import main

    out = tmp_path / "noisy_eval.json"
    assert main(
        [
            "make-noisy-eval",
            "--study",
            str(study_path),
            "--experiment-id",
            "TS12-E01",
            "--prepared-root",
            str(prepared),
            "--split-registry",
            str(split_path),
            "--pair-manifest",
            str(manifest_path),
            "--artifact-id",
            "NOISY-VAL-v1",
            "--out",
            str(out),
        ]
    ) == 0
    payload = __import__("json").loads(out.read_text(encoding="utf-8"))
    assert payload["artifact_id"] == "NOISY-VAL-v1"
    assert payload["schema_version"] == "dluxshera_ml_noisy_eval_recipe/1"
    assert payload["underlying_pair_manifest"]["content_sha256"] == manifest.manifest["content_identity"]["sha256"]
    assert payload["record_count"] == len(manifest.records)
    assert payload["content_identity"]["sha256"]
    for index, record in enumerate(payload["records"]):
        assert record["seeds"] == pair_noise_side_seeds(
            payload["noise_model"],
            pair_record_id=record["pair_record_id"],
            dynamic_seed_offset=index,
        )


def test_s12_contract_rejects_missing_or_wrong_noisy_eval_artifact(tmp_path: Path) -> None:
    prepared = _write_prepared_fixture(tmp_path / "prepared")
    catalog = load_sample_catalog(prepared)
    registry = generate_split_registry(
        catalog,
        seed=7,
        science_fractions={"train": 1.0, "validation": 0.0, "test": 0.0},
        nuisance_fractions={"train": 1.0, "validation": 0.0, "test": 0.0},
    )
    split_path = tmp_path / "split.json"
    write_split_registry(split_path, registry)
    policy = PairPolicy(
        policy_id="fixture_policy",
        family_weights={"A": 1.0},
        same_pair_id=True,
        min_fisher_distance=0.5,
        max_fisher_distance=3.0,
        include_reverse=False,
        max_sampling_attempts=4000,
    )
    manifest = generate_frozen_pair_manifest(
        catalog,
        registry,
        policy=policy,
        artifact_id="FIXTURE-VALIDATION-v1",
        split="validation",
        seed=4,
        pairs_per_slice=2,
        eval_slices={"fixture": {"science_split": "train", "nuisance_split": "train"}},
    )
    manifest_path = tmp_path / "pairs"
    write_pair_manifest(manifest_path, manifest)
    study_path = tmp_path / "study.yaml"
    study_path.write_text(
        """
study_id: TS12
dataset: {artifact_id: PREP-V3-v1}
split_registry: {artifact_id: SPLIT-ML-v1}
auxiliary_artifacts:
  noisy_validation_recipe:
    artifact_id: NOISY-VAL-v1
    schema_version: dluxshera_ml_noisy_eval_recipe/1
    underlying_validation_artifact: FIXTURE-VALIDATION-v1
pair_policies:
  fixture_policy:
    family_weights: {A: 1.0}
    same_pair_id: true
    min_fisher_distance: 0.5
    max_fisher_distance: 3.0
    include_reverse: false
    max_sampling_attempts: 4000
evaluation_artifacts:
  validation:
    artifact_id: FIXTURE-VALIDATION-v1
    pair_policy_id: fixture_policy
    split: validation
    seed: 4
    pairs_per_slice: 2
    eval_slices: {fixture: {science_split: train, nuisance_split: train}}
defaults:
  pair_policy_id: fixture_policy
  validation_artifact: validation
  require_frozen_validation_manifest: true
  evaluate_test: false
  validation_noise:
    enabled: true
    apply_to: observation
    noise_model: shera_observation
    photon_noise: true
    read_noise: false
    dark_current: false
    seed: 12031
    training_dynamic: false
    negative_policy: clip
experiments:
  TS12-E01:
    seeds: [11]
""",
        encoding="utf-8",
    )
    study = load_study_prescription(study_path)
    config = resolve_study_experiment_config(study, experiment_id="TS12-E01")
    with pytest.raises(ValueError, match="noisy-eval-artifact"):
        load_study_contract_artifacts(
            study=study,
            prepared_root=prepared,
            split_registry_path=split_path,
            validation_manifest_path=manifest_path,
            experiment_id="TS12-E01",
            config=config,
        )

    from work.experiments.ml.materialize_study_artifacts import main

    noisy_path = tmp_path / "noisy_eval.json"
    assert main(
        [
            "make-noisy-eval",
            "--study",
            str(study_path),
            "--experiment-id",
            "TS12-E01",
            "--prepared-root",
            str(prepared),
            "--split-registry",
            str(split_path),
            "--pair-manifest",
            str(manifest_path),
            "--artifact-id",
            "WRONG-NOISY-v1",
            "--out",
            str(noisy_path),
        ]
    ) == 0
    with pytest.raises(ValueError, match="artifact_id"):
        load_study_contract_artifacts(
            study=study,
            prepared_root=prepared,
            split_registry_path=split_path,
            validation_manifest_path=manifest_path,
            noisy_eval_artifact_path=noisy_path,
            experiment_id="TS12-E01",
            config=config,
        )
