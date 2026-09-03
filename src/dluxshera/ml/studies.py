from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from .catalog import SampleCatalog, load_sample_catalog
from .pairs import (
    PairManifest,
    PairPolicy,
    load_pair_manifest,
    pair_manifest_content_hash,
)
from .splits import SplitRegistry, load_split_registry, split_registry_content_sha256

__all__ = [
    "load_study_prescription",
    "load_study_contract_artifacts",
    "resolve_study_experiment_config",
    "validate_experiment_policy_for_study",
    "validate_evaluation_artifact_against_recipe",
    "validate_prepared_dataset_for_study",
    "validate_split_registry_for_study",
    "validate_study_contract",
]


def load_study_prescription(path: Path) -> dict[str, Any]:
    """Load a tracked ML study prescription from YAML or JSON."""
    text = Path(path).read_text(encoding="utf-8")
    if str(path).endswith((".yaml", ".yml")):
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Study YAML files require PyYAML.") from exc
        payload = yaml.safe_load(text)
    else:
        import json

        payload = json.loads(text)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a mapping.")
    return dict(payload)


def _policy_mapping(study: Mapping[str, Any], policy_id: str) -> dict[str, Any]:
    policies = study.get("pair_policies")
    if not isinstance(policies, Mapping):
        raise ValueError("Study prescription must define pair_policies.")
    if policy_id not in policies:
        raise ValueError(f"Unknown pair_policy_id {policy_id!r}.")
    policy = dict(policies[policy_id])
    policy.setdefault("policy_id", policy_id)
    if str(policy["policy_id"]) != str(policy_id):
        raise ValueError(
            f"Policy key {policy_id!r} contains mismatched policy_id {policy['policy_id']!r}."
        )
    return PairPolicy.from_dict(policy).to_dict()


def _experiment_mapping(study: Mapping[str, Any], experiment_id: str) -> dict[str, Any]:
    experiments = study.get("experiments")
    if not isinstance(experiments, Mapping):
        raise ValueError("Study prescription must define experiments.")
    if experiment_id not in experiments:
        raise ValueError(f"Unknown experiment_id {experiment_id!r}.")
    experiment = copy.deepcopy(experiments[experiment_id])
    if not isinstance(experiment, Mapping):
        raise ValueError(f"Experiment {experiment_id!r} must be a mapping.")
    return dict(experiment)


def resolve_study_experiment_config(
    study: Mapping[str, Any],
    *,
    experiment_id: str,
    run_id: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Resolve one study experiment into the mapping consumed by training."""
    experiment = _experiment_mapping(study, experiment_id)
    policy_id = str(experiment.pop("pair_policy_id"))
    config = copy.deepcopy(experiment)
    config["study_id"] = str(study["study_id"])
    config["experiment_id"] = str(experiment_id)
    config["run_id"] = str(run_id or config.get("run_id"))
    config["pair_policy"] = _policy_mapping(study, policy_id)
    config["dataset"] = copy.deepcopy(study.get("dataset", {}))
    if device is not None:
        config["device"] = str(device)
    return config


def validate_prepared_dataset_for_study(catalog: SampleCatalog, study: Mapping[str, Any]) -> None:
    """Reject a prepared dataset whose identity differs from the prescription."""
    dataset = study.get("dataset", {})
    if not isinstance(dataset, Mapping):
        raise ValueError("Study prescription dataset must be a mapping.")
    expected_artifact_id = dataset.get("artifact_id")
    if expected_artifact_id and str(expected_artifact_id) != str(catalog.artifact_id):
        raise ValueError(
            "Prepared dataset artifact_id does not match study prescription "
            f"({expected_artifact_id} != {catalog.artifact_id})."
        )
    expected_hash = dataset.get("prepared_dataset_hash")
    if expected_hash and str(expected_hash) != str(catalog.prepared_dataset_hash):
        raise ValueError(
            "Prepared dataset hash does not match study prescription "
            f"({expected_hash} != {catalog.prepared_dataset_hash})."
        )


def _split_expectation(study: Mapping[str, Any]) -> dict[str, Any]:
    split_registry = study.get("split_registry")
    if isinstance(split_registry, Mapping):
        return dict(split_registry)
    dataset = study.get("dataset", {})
    if isinstance(dataset, Mapping) and dataset.get("split_registry_id"):
        return {"artifact_id": dataset.get("split_registry_id")}
    raise ValueError("Study prescription must define split_registry.artifact_id.")


def validate_split_registry_for_study(
    split_registry: SplitRegistry,
    study: Mapping[str, Any],
) -> None:
    """Reject a split registry whose stable identity differs from the prescription."""
    expected = _split_expectation(study)
    expected_artifact_id = expected.get("artifact_id")
    if expected_artifact_id and str(expected_artifact_id) != split_registry.artifact_id:
        raise ValueError(
            "Split registry artifact_id does not match study prescription "
            f"({expected_artifact_id} != {split_registry.artifact_id})."
        )
    expected_content = expected.get("content_sha256")
    actual_content = split_registry_content_sha256(split_registry)
    if expected_content and str(expected_content) != actual_content:
        raise ValueError(
            "Split registry content_sha256 does not match study prescription "
            f"({expected_content} != {actual_content})."
        )
    dataset = study.get("dataset", {})
    if isinstance(dataset, Mapping):
        expected_hash = dataset.get("prepared_dataset_hash")
        actual_hash = split_registry.prepared_dataset.get("prepared_dataset_hash")
        if expected_hash and str(expected_hash) != str(actual_hash):
            raise ValueError(
                "Split registry prepared dataset hash does not match study prescription "
                f"({expected_hash} != {actual_hash})."
            )
        expected_dataset_artifact = dataset.get("artifact_id")
        actual_dataset_artifact = split_registry.prepared_dataset.get("artifact_id")
        if expected_dataset_artifact and str(expected_dataset_artifact) != str(
            actual_dataset_artifact
        ):
            raise ValueError(
                "Split registry prepared artifact_id does not match study prescription "
                f"({expected_dataset_artifact} != {actual_dataset_artifact})."
            )


def _recipe(study: Mapping[str, Any], artifact_key: str) -> dict[str, Any]:
    artifacts = study.get("evaluation_artifacts")
    if not isinstance(artifacts, Mapping) or artifact_key not in artifacts:
        raise ValueError(f"Study prescription has no evaluation_artifacts.{artifact_key}.")
    recipe = dict(artifacts[artifact_key])
    if "pair_policy_id" not in recipe:
        raise ValueError(f"Evaluation artifact {artifact_key!r} must define pair_policy_id.")
    return recipe


def _manifest_content_sha256(pair_manifest: PairManifest) -> str:
    return pair_manifest_content_hash(pair_manifest.manifest, pair_manifest.records)


def validate_evaluation_artifact_against_recipe(
    pair_manifest: PairManifest,
    *,
    study: Mapping[str, Any],
    artifact_key: str,
    split_registry: SplitRegistry,
) -> None:
    """Validate a frozen manifest against a study-level evaluation recipe."""
    recipe = _recipe(study, artifact_key)
    policy = _policy_mapping(study, str(recipe["pair_policy_id"]))
    manifest = pair_manifest.manifest
    expected = {
        "artifact_id": str(recipe.get("artifact_id", pair_manifest.artifact_id)),
        "split": str(recipe["split"]),
        "seed": int(recipe["seed"]),
        "pairs_per_slice_requested": int(recipe["pairs_per_slice"]),
        "eval_slices": {str(k): dict(v) for k, v in dict(recipe["eval_slices"]).items()},
        "pair_policy": policy,
    }
    actual = {
        "artifact_id": str(manifest.get("artifact_id")),
        "split": str(manifest.get("split")),
        "seed": int(manifest.get("seed")),
        "pairs_per_slice_requested": int(manifest.get("pairs_per_slice_requested")),
        "eval_slices": {str(k): dict(v) for k, v in dict(manifest.get("eval_slices", {})).items()},
        "pair_policy": dict(manifest.get("pair_policy", {})),
    }
    if actual != expected:
        raise ValueError(
            f"Frozen {artifact_key} manifest does not match study recipe: "
            f"expected={expected}, actual={actual}."
        )
    dataset = study.get("dataset", {})
    if isinstance(dataset, Mapping):
        prepared = manifest.get("prepared_dataset", {})
        expected_dataset_artifact = dataset.get("artifact_id")
        actual_dataset_artifact = prepared.get("artifact_id")
        if expected_dataset_artifact and str(expected_dataset_artifact) != str(
            actual_dataset_artifact
        ):
            raise ValueError(
                f"Frozen {artifact_key} manifest prepared artifact_id does not match study "
                f"({expected_dataset_artifact} != {actual_dataset_artifact})."
            )
        expected_dataset_hash = dataset.get("prepared_dataset_hash")
        actual_dataset_hash = prepared.get("prepared_dataset_hash")
        if expected_dataset_hash and str(expected_dataset_hash) != str(actual_dataset_hash):
            raise ValueError(
                f"Frozen {artifact_key} manifest prepared dataset hash does not match study "
                f"({expected_dataset_hash} != {actual_dataset_hash})."
            )
    split_identity = manifest.get("split_registry", {})
    if split_identity.get("artifact_id") != split_registry.artifact_id:
        raise ValueError(
            f"Frozen {artifact_key} manifest split registry "
            f"{split_identity.get('artifact_id')!r} does not match {split_registry.artifact_id!r}."
        )
    expected_split_hash = split_identity.get("content_sha256")
    actual_split_hash = split_registry_content_sha256(split_registry)
    if expected_split_hash and expected_split_hash != actual_split_hash:
        raise ValueError(
            f"Frozen {artifact_key} manifest split content hash does not match current split "
            f"({expected_split_hash} != {actual_split_hash})."
        )
    study_split_hash = _split_expectation(study).get("content_sha256")
    if study_split_hash and str(study_split_hash) != actual_split_hash:
        raise ValueError(
            f"Frozen {artifact_key} manifest split content hash does not match study "
            f"({study_split_hash} != {actual_split_hash})."
        )
    content_identity = manifest.get("content_identity", {})
    if not isinstance(content_identity, Mapping) or not content_identity.get("sha256"):
        raise ValueError(f"Frozen {artifact_key} manifest is missing content_identity.sha256.")
    actual_manifest_hash = _manifest_content_sha256(pair_manifest)
    if str(content_identity["sha256"]) != actual_manifest_hash:
        raise ValueError(
            f"Frozen {artifact_key} manifest content_identity.sha256 does not match content "
            f"({content_identity['sha256']} != {actual_manifest_hash})."
        )
    expected_manifest_hash = recipe.get("content_sha256")
    if expected_manifest_hash and str(expected_manifest_hash) != actual_manifest_hash:
        raise ValueError(
            f"Frozen {artifact_key} manifest content_sha256 does not match study recipe "
            f"({expected_manifest_hash} != {actual_manifest_hash})."
        )


def validate_experiment_policy_for_study(
    study: Mapping[str, Any],
    *,
    experiment_id: str,
    config: Mapping[str, Any] | None = None,
) -> None:
    """Validate that an experiment resolves to its study-level pair policy."""
    expected = resolve_study_experiment_config(study, experiment_id=experiment_id)["pair_policy"]
    if config is None:
        return
    actual = PairPolicy.from_dict(config.get("pair_policy", {})).to_dict()
    if actual != expected:
        raise ValueError(
            "Experiment pair_policy does not match study prescription "
            f"for {experiment_id}: expected={expected}, actual={actual}."
        )


def validate_study_contract(
    *,
    study: Mapping[str, Any],
    catalog: SampleCatalog,
    split_registry: SplitRegistry,
    validation_manifest: PairManifest | None = None,
    test_manifest: PairManifest | None = None,
    experiment_id: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> None:
    """Validate the loaded artifacts that define one study run."""
    validate_prepared_dataset_for_study(catalog, study)
    split_registry.validate_catalog(catalog)
    validate_split_registry_for_study(split_registry, study)
    if experiment_id is not None:
        validate_experiment_policy_for_study(study, experiment_id=experiment_id, config=config)
    experiment = config or (
        resolve_study_experiment_config(study, experiment_id=experiment_id)
        if experiment_id is not None
        else {}
    )
    validation_key = str(experiment.get("validation_artifact", "validation"))
    test_key = str(experiment.get("test_artifact", "test"))
    if (
        bool(experiment.get("require_frozen_validation_manifest", False))
        and validation_manifest is None
    ):
        raise ValueError("Study contract requires an explicit frozen validation manifest.")
    if validation_manifest is not None:
        validate_evaluation_artifact_against_recipe(
            validation_manifest,
            study=study,
            artifact_key=validation_key,
            split_registry=split_registry,
        )
    if test_manifest is not None:
        validate_evaluation_artifact_against_recipe(
            test_manifest,
            study=study,
            artifact_key=test_key,
            split_registry=split_registry,
        )


def load_study_contract_artifacts(
    *,
    study: Mapping[str, Any],
    prepared_root: Path,
    split_registry_path: Path,
    validation_manifest_path: Path | None = None,
    test_manifest_path: Path | None = None,
    experiment_id: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and validate catalog, split registry, and optional frozen manifests."""
    catalog = load_sample_catalog(prepared_root)
    split_registry = load_split_registry(split_registry_path, catalog=catalog)
    validation_manifest = (
        None
        if validation_manifest_path is None
        else load_pair_manifest(
            validation_manifest_path,
            catalog=catalog,
            split_registry=split_registry,
        )
    )
    test_manifest = (
        None
        if test_manifest_path is None
        else load_pair_manifest(
            test_manifest_path,
            catalog=catalog,
            split_registry=split_registry,
        )
    )
    validate_study_contract(
        study=study,
        catalog=catalog,
        split_registry=split_registry,
        validation_manifest=validation_manifest,
        test_manifest=test_manifest,
        experiment_id=experiment_id,
        config=config,
    )
    return {
        "catalog": catalog,
        "split_registry": split_registry,
        "validation_manifest": validation_manifest,
        "test_manifest": test_manifest,
    }
