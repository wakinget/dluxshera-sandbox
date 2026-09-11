from __future__ import annotations

import copy
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .artifacts import load_artifact_lock, validate_artifact_lock
from .catalog import SampleCatalog, load_sample_catalog
from .pairs import (
    PairManifest,
    PairPolicy,
    load_pair_manifest,
    pair_manifest_content_hash,
)
from .noisy_eval import load_noisy_eval_recipe, validate_noisy_eval_recipe
from .splits import SplitRegistry, load_split_registry, split_registry_content_sha256
from .scaling import load_intensity_scaler

__all__ = [
    "StudyRunPlanRow",
    "expand_study_run_plan",
    "load_study_prescription",
    "load_study_contract_artifacts",
    "resolve_study_artifact_contract",
    "resolve_study_experiment_config",
    "write_study_run_plan",
    "validate_experiment_policy_for_study",
    "validate_evaluation_artifact_against_recipe",
    "validate_noisy_eval_artifact_for_study",
    "validate_prepared_dataset_for_study",
    "validate_split_registry_for_study",
    "validate_study_contract",
]


def _stable_sha256(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StudyRunPlanRow:
    """Describe one concrete planned ML training execution."""

    study_id: str
    experiment_id: str
    run_id: str
    seed: int
    enabled: bool
    model_config_identity: str
    pair_policy_id: str
    pair_policy_identity: str
    dataset_selection_identity: str
    training_identity: str
    validation_artifact: str | None
    test_artifact: str | None
    artifact_profile: str | None
    split_registry_artifact_id: str | None
    scaler_artifact_id: str | None
    audit_artifacts: tuple[str, ...]
    auxiliary_artifacts: Mapping[str, Any]
    shared_reference_runs: tuple[str, ...]
    science_loss_identity: str
    pair_consistency_identity: str
    noise_identity: str
    validation_noise_identity: str
    artifact_lock_id: str | None
    output_root: str | None
    hpc_profile: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/CSV-ready run-plan row."""
        return {
            "study_id": self.study_id,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "seed": int(self.seed),
            "enabled": bool(self.enabled),
            "model_config_identity": self.model_config_identity,
            "pair_policy_id": self.pair_policy_id,
            "pair_policy_identity": self.pair_policy_identity,
            "dataset_selection_identity": self.dataset_selection_identity,
            "training_identity": self.training_identity,
            "validation_artifact": self.validation_artifact,
            "test_artifact": self.test_artifact,
            "artifact_profile": self.artifact_profile,
            "split_registry_artifact_id": self.split_registry_artifact_id,
            "scaler_artifact_id": self.scaler_artifact_id,
            "audit_artifacts": list(self.audit_artifacts),
            "auxiliary_artifacts": dict(self.auxiliary_artifacts),
            "shared_reference_runs": list(self.shared_reference_runs),
            "science_loss_identity": self.science_loss_identity,
            "pair_consistency_identity": self.pair_consistency_identity,
            "noise_identity": self.noise_identity,
            "validation_noise_identity": self.validation_noise_identity,
            "artifact_lock_id": self.artifact_lock_id,
            "output_root": self.output_root,
            "hpc_profile": self.hpc_profile,
        }


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


def _deep_update(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_update(dict(out[key]), value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _default_artifact_profile(study: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "profile_id": "default",
        "split_registry": copy.deepcopy(dict(study.get("split_registry", {}) or {})),
        "scaler": copy.deepcopy(dict(study.get("scaler", {}) or {})),
        "artifact_lock": copy.deepcopy(dict(study.get("artifact_lock", {}) or {})),
    }


def _artifact_profile_mapping(study: Mapping[str, Any], profile_id: str | None) -> dict[str, Any]:
    profiles = study.get("artifact_profiles")
    if profile_id in (None, ""):
        if isinstance(profiles, Mapping) and "standard" in profiles:
            profile_id = "standard"
        else:
            return _default_artifact_profile(study)
    if not isinstance(profiles, Mapping) or str(profile_id) not in profiles:
        if str(profile_id) == "default":
            return _default_artifact_profile(study)
        valid = sorted(str(key) for key in dict(profiles or {}))
        raise ValueError(f"Unknown artifact_profile {profile_id!r}; valid profiles: {valid}.")
    profile = copy.deepcopy(dict(profiles[str(profile_id)]))
    profile.setdefault("profile_id", str(profile_id))
    return profile


def resolve_study_artifact_contract(
    study: Mapping[str, Any],
    profile_id: str | None = None,
) -> dict[str, Any]:
    profile = _artifact_profile_mapping(study, profile_id or None)
    split_registry = _deep_update(
        dict(study.get("split_registry", {}) or {}),
        dict(profile.get("split_registry", {}) or {}),
    )
    scaler = _deep_update(
        dict(study.get("scaler", {}) or {}),
        dict(profile.get("scaler", {}) or {}),
    )
    artifact_lock = _deep_update(
        dict(study.get("artifact_lock", {}) or {}),
        dict(profile.get("artifact_lock", {}) or {}),
    )
    return {
        "profile_id": str(profile.get("profile_id", profile_id or "default")),
        "split_registry": split_registry,
        "scaler": scaler,
        "artifact_lock": artifact_lock,
    }


def _resolved_artifact_contract(
    study: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return resolve_study_artifact_contract(
        study,
        None if config.get("artifact_profile") in (None, "") else str(config.get("artifact_profile")),
    )


def _resolved_experiment_mapping(study: Mapping[str, Any], experiment_id: str) -> dict[str, Any]:
    experiment = _experiment_mapping(study, experiment_id)
    config: dict[str, Any] = {}
    defaults = study.get("defaults")
    if isinstance(defaults, Mapping):
        config = _deep_update(config, defaults)
    presets = study.get("presets", {})
    inherited = experiment.pop("inherits", [])
    if isinstance(inherited, str):
        inherited = [inherited]
    for preset_name in inherited or []:
        if not isinstance(presets, Mapping) or str(preset_name) not in presets:
            raise ValueError(f"Experiment {experiment_id!r} inherits unknown preset {preset_name!r}.")
        preset = presets[str(preset_name)]
        if not isinstance(preset, Mapping):
            raise ValueError(f"Study preset {preset_name!r} must be a mapping.")
        config = _deep_update(config, preset)
    return _deep_update(config, experiment)


def _run_rows_for_experiment(
    experiment: Mapping[str, Any],
    *,
    experiment_id: str,
) -> list[dict[str, Any]]:
    if "runs" in experiment:
        rows = []
        for idx, row in enumerate(experiment.get("runs") or [], start=1):
            if not isinstance(row, Mapping):
                raise ValueError(f"{experiment_id}.runs[{idx}] must be a mapping.")
            run = dict(row)
            run.setdefault("run_id", f"{experiment_id}-R{idx:03d}")
            if "seed" not in run:
                raise ValueError(f"{experiment_id}.runs[{idx}] is missing seed.")
            rows.append(run)
        return rows
    if "seeds" in experiment:
        return [
            {"run_id": f"{experiment_id}-R{idx:03d}", "seed": int(seed)}
            for idx, seed in enumerate(experiment.get("seeds") or [], start=1)
        ]
    return [
        {
            "run_id": experiment.get("run_id", f"{experiment_id}-R001"),
            "seed": int(experiment.get("seed", 0)),
        }
    ]


def _select_run_overrides(
    experiment: Mapping[str, Any],
    *,
    experiment_id: str,
    run_id: str | None,
) -> dict[str, Any]:
    rows = _run_rows_for_experiment(experiment, experiment_id=experiment_id)
    if run_id is None:
        return dict(rows[0])
    for row in rows:
        if str(row["run_id"]) == str(run_id):
            return dict(row)
    raise ValueError(f"Unknown run_id {run_id!r} for experiment {experiment_id!r}.")


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
    experiment = _resolved_experiment_mapping(study, experiment_id)
    run_overrides = _select_run_overrides(
        experiment,
        experiment_id=experiment_id,
        run_id=run_id,
    )
    experiment.pop("seeds", None)
    experiment.pop("runs", None)
    policy_id = str(experiment.pop("pair_policy_id"))
    config = copy.deepcopy(experiment)
    config = _deep_update(config, run_overrides)
    config["study_id"] = str(study["study_id"])
    config["experiment_id"] = str(experiment_id)
    config["run_id"] = str(config.get("run_id"))
    config["seed"] = int(config.get("seed", 0))
    config["pair_policy"] = _policy_mapping(study, policy_id)
    config["dataset"] = copy.deepcopy(study.get("dataset", {}))
    config["auxiliary_artifacts"] = _deep_update(
        dict(study.get("auxiliary_artifacts", {}) or {}),
        dict(config.get("auxiliary_artifacts", {}) or {}),
    )
    if "shared_reference" in study and "shared_reference" not in config:
        config["shared_reference"] = copy.deepcopy(study.get("shared_reference"))
    contract = _resolved_artifact_contract(study, config)
    config["artifact_profile"] = contract["profile_id"]
    config["artifact_contract"] = contract
    split_cfg = contract.get("split_registry", {})
    scaler_cfg = contract.get("scaler", {})
    lock_cfg = contract.get("artifact_lock", {})
    config["split_registry_artifact_id"] = (
        split_cfg.get("artifact_id") if isinstance(split_cfg, Mapping) else None
    )
    config["scaler_artifact_id"] = (
        scaler_cfg.get("artifact_id") if isinstance(scaler_cfg, Mapping) else None
    )
    config["artifact_lock_id"] = (
        lock_cfg.get("lock_id") if isinstance(lock_cfg, Mapping) else None
    )
    config["primary_validation_artifact_key"] = config.get("validation_artifact")
    config["test_artifact_key"] = config.get("test_artifact")
    config["declared_audit_artifact_keys"] = [
        str(key) for key in config.get("audit_artifacts", []) or []
    ]
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


def expand_study_run_plan(
    study: Mapping[str, Any],
    *,
    experiment_ids: Sequence[str] | None = None,
    run_ids: Sequence[str] | None = None,
    include_disabled: bool = False,
    output_root: Path | str | None = None,
) -> list[StudyRunPlanRow]:
    """Expand a study prescription into deterministic concrete run rows."""
    experiments = study.get("experiments")
    if not isinstance(experiments, Mapping):
        raise ValueError("Study prescription must define experiments.")
    selected_experiments = (
        [str(value) for value in experiment_ids]
        if experiment_ids is not None
        else sorted(str(value) for value in experiments)
    )
    selected_runs = None if run_ids is None else {str(value) for value in run_ids}
    rows: list[StudyRunPlanRow] = []
    for experiment_id in selected_experiments:
        experiment = _resolved_experiment_mapping(study, experiment_id)
        enabled = bool(experiment.get("enabled", True))
        for run in _run_rows_for_experiment(experiment, experiment_id=experiment_id):
            run_id = str(run["run_id"])
            if selected_runs is not None and run_id not in selected_runs:
                continue
            config = resolve_study_experiment_config(
                study,
                experiment_id=experiment_id,
                run_id=run_id,
            )
            row_enabled = bool(config.get("enabled", enabled))
            if not row_enabled and not include_disabled:
                continue
            pair_policy = config["pair_policy"]
            dataset_selection = {
                "dataset": config.get("dataset", {}),
                "training_data": config.get("training_data", {}),
                "pair_policy_dataset_fields": {
                    key: pair_policy.get(key)
                    for key in (
                        "dataset_families",
                        "dataset_family_weights",
                        "dataset_family_sampling",
                        "joint_train_prefix_size",
                    )
                    if pair_policy.get(key) not in (None, [], {})
                },
            }
            output_root_value = (
                None
                if output_root is None
                else str(Path(output_root) / str(study["study_id"]) / experiment_id / run_id)
            )
            artifact_contract = dict(config.get("artifact_contract", {}))
            lock_cfg = artifact_contract.get("artifact_lock", {})
            split_cfg = artifact_contract.get("split_registry", {})
            scaler_cfg = artifact_contract.get("scaler", {})
            hpc_cfg = config.get("hpc", {})
            auxiliary_artifacts = _deep_update(
                dict(study.get("auxiliary_artifacts", {}) or {}),
                dict(config.get("auxiliary_artifacts", {}) or {}),
            )
            rows.append(
                StudyRunPlanRow(
                    study_id=str(study["study_id"]),
                    experiment_id=experiment_id,
                    run_id=run_id,
                    seed=int(config["seed"]),
                    enabled=row_enabled,
                    model_config_identity=_stable_sha256(config.get("model", {})),
                    pair_policy_id=str(pair_policy.get("policy_id")),
                    pair_policy_identity=_stable_sha256(pair_policy),
                    dataset_selection_identity=_stable_sha256(dataset_selection),
                    training_identity=_stable_sha256(config.get("training", {})),
                    validation_artifact=config.get("validation_artifact"),
                    test_artifact=config.get("test_artifact"),
                    artifact_profile=config.get("artifact_profile"),
                    split_registry_artifact_id=split_cfg.get("artifact_id")
                    if isinstance(split_cfg, Mapping)
                    else None,
                    scaler_artifact_id=scaler_cfg.get("artifact_id")
                    if isinstance(scaler_cfg, Mapping)
                    else None,
                    audit_artifacts=tuple(str(v) for v in config.get("audit_artifacts", []) or []),
                    auxiliary_artifacts=auxiliary_artifacts,
                    shared_reference_runs=tuple(
                        str(v) for v in config.get("shared_reference_runs", []) or []
                    ),
                    science_loss_identity=_stable_sha256(config.get("science_loss", {})),
                    pair_consistency_identity=_stable_sha256(config.get("pair_consistency", {})),
                    noise_identity=_stable_sha256(config.get("noise", {})),
                    validation_noise_identity=_stable_sha256(config.get("validation_noise", {})),
                    artifact_lock_id=lock_cfg.get("lock_id")
                    if isinstance(lock_cfg, Mapping)
                    else None,
                    output_root=output_root_value,
                    hpc_profile=hpc_cfg.get("profile") if isinstance(hpc_cfg, Mapping) else None,
                )
            )
    return rows


def write_study_run_plan(
    *,
    study: Mapping[str, Any],
    output_dir: Path,
    rows: Sequence[StudyRunPlanRow],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a deterministic study run plan as CSV plus JSON summary."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} exists and is non-empty; pass overwrite=True.")
    output_dir.mkdir(parents=True, exist_ok=True)
    row_payloads = [row.to_dict() for row in rows]
    csv_path = output_dir / "run_plan.csv"
    json_path = output_dir / "run_plan.json"
    if row_payloads:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row_payloads[0]))
            writer.writeheader()
            writer.writerows(row_payloads)
    else:
        csv_path.write_text("", encoding="utf-8")
    summary = {
        "schema_version": "dluxshera_ml_run_plan/1",
        "study_id": study.get("study_id"),
        "run_count": len(row_payloads),
        "enabled_run_count": sum(1 for row in rows if row.enabled),
        "experiment_counts": {
            experiment_id: sum(1 for row in rows if row.experiment_id == experiment_id)
            for experiment_id in sorted({row.experiment_id for row in rows})
        },
        "rows": row_payloads,
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "run_count": len(row_payloads),
        "enabled_run_count": summary["enabled_run_count"],
    }


def _split_expectation(
    study: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if config is not None:
        contract = config.get("artifact_contract")
        if isinstance(contract, Mapping):
            split_registry = contract.get("split_registry")
            if isinstance(split_registry, Mapping):
                return dict(split_registry)
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
    *,
    config: Mapping[str, Any] | None = None,
) -> None:
    """Reject a split registry whose stable identity differs from the prescription."""
    expected = _split_expectation(study, config=config)
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
    config: Mapping[str, Any] | None = None,
) -> None:
    """Validate a frozen manifest against a study-level evaluation recipe."""
    recipe = _recipe(study, artifact_key)
    if config is not None and recipe.get("artifact_profile") not in (None, ""):
        actual_profile = config.get("artifact_profile")
        if actual_profile not in (None, "") and str(recipe["artifact_profile"]) != str(actual_profile):
            raise ValueError(
                f"Evaluation artifact {artifact_key!r} belongs to artifact profile "
                f"{recipe['artifact_profile']!r}, but the run resolved profile "
                f"{actual_profile!r}."
            )
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
        "pair_policy": PairPolicy.from_dict(dict(manifest.get("pair_policy", {}))).to_dict(),
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
    study_split_hash = _split_expectation(study, config=config).get("content_sha256")
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
    expected_count = recipe.get("ordered_pair_count")
    if expected_count is not None and int(expected_count) != len(pair_manifest.records):
        raise ValueError(
            f"Frozen {artifact_key} manifest ordered_pair_count does not match study recipe "
            f"({expected_count} != {len(pair_manifest.records)})."
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
    artifact_lock_path: Path | None = None,
    scaler_path: Path | None = None,
    noisy_eval_artifact_path: Path | None = None,
    audit_manifests: Mapping[str, PairManifest] | None = None,
    experiment_id: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> None:
    """Validate the loaded artifacts that define one study run."""
    validate_prepared_dataset_for_study(catalog, study)
    split_registry.validate_catalog(catalog)
    validate_split_registry_for_study(split_registry, study, config=config)
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
            config=experiment,
        )
    validate_noisy_eval_artifact_for_study(
        study=study,
        config=experiment,
        catalog=catalog,
        split_registry=split_registry,
        validation_manifest=validation_manifest,
        noisy_eval_artifact_path=noisy_eval_artifact_path,
    )
    if test_manifest is not None:
        validate_evaluation_artifact_against_recipe(
            test_manifest,
            study=study,
            artifact_key=test_key,
            split_registry=split_registry,
            config=experiment,
        )
    declared_audits = {str(key) for key in experiment.get("audit_artifacts", []) or []}
    supplied_audits = {str(key) for key in dict(audit_manifests or {})}
    missing_audits = sorted(declared_audits - supplied_audits)
    if missing_audits:
        raise ValueError(
            "Study contract requires audit manifest(s) declared by the experiment: "
            f"{missing_audits}."
        )
    for artifact_key, manifest in dict(audit_manifests or {}).items():
        validate_evaluation_artifact_against_recipe(
            manifest,
            study=study,
            artifact_key=str(artifact_key),
            split_registry=split_registry,
            config=experiment,
        )
    lock_cfg = (
        experiment.get("artifact_contract", {}).get("artifact_lock", {})
        if isinstance(experiment.get("artifact_contract"), Mapping)
        else study.get("artifact_lock", {})
    )
    lock_required = isinstance(lock_cfg, Mapping) and bool(lock_cfg.get("required", False))
    if lock_required and artifact_lock_path is None:
        raise ValueError("Study contract requires an artifact lock for production execution.")
    if artifact_lock_path is not None:
        lock = load_artifact_lock(artifact_lock_path)
        scaler = None if scaler_path is None else load_intensity_scaler(scaler_path)
        manifests: dict[str, PairManifest] = {}
        if validation_manifest is not None:
            manifests[validation_key] = validation_manifest
        if test_manifest is not None:
            manifests[test_key] = test_manifest
        manifests.update(dict(audit_manifests or {}))
        validate_artifact_lock(
            lock=lock,
            study=study,
            catalog=catalog,
            split_registry=split_registry,
            scaler=scaler,
            pair_manifests=manifests,
        )


def validate_noisy_eval_artifact_for_study(
    *,
    study: Mapping[str, Any],
    config: Mapping[str, Any],
    catalog: SampleCatalog,
    split_registry: SplitRegistry,
    validation_manifest: PairManifest | None,
    noisy_eval_artifact_path: Path | None,
) -> dict[str, Any] | None:
    """Validate an optional fixed noisy-validation runtime recipe."""
    aux = config.get("auxiliary_artifacts", study.get("auxiliary_artifacts", {}))
    expected = {}
    if isinstance(aux, Mapping):
        expected = dict(aux.get("noisy_validation_recipe", {}) or {})
    if not expected:
        return None
    if validation_manifest is None:
        raise ValueError("Noisy-validation recipe validation requires a frozen validation manifest.")
    if noisy_eval_artifact_path is None:
        raise ValueError("Study contract requires --noisy-eval-artifact for fixed noisy validation.")
    recipe = load_noisy_eval_recipe(noisy_eval_artifact_path)
    return validate_noisy_eval_recipe(
        recipe,
        study=study,
        config=config,
        catalog=catalog,
        split_registry=split_registry,
        validation_manifest=validation_manifest,
        expected=expected,
    )


def load_study_contract_artifacts(
    *,
    study: Mapping[str, Any],
    prepared_root: Path,
    split_registry_path: Path,
    validation_manifest_path: Path | None = None,
    test_manifest_path: Path | None = None,
    artifact_lock_path: Path | None = None,
    scaler_path: Path | None = None,
    noisy_eval_artifact_path: Path | None = None,
    audit_manifest_paths: Mapping[str, Path] | None = None,
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
    audit_manifests = {
        str(key): load_pair_manifest(
            Path(path),
            catalog=catalog,
            split_registry=split_registry,
        )
        for key, path in dict(audit_manifest_paths or {}).items()
    }
    validate_study_contract(
        study=study,
        catalog=catalog,
        split_registry=split_registry,
        validation_manifest=validation_manifest,
        test_manifest=test_manifest,
        artifact_lock_path=artifact_lock_path,
        scaler_path=scaler_path,
        noisy_eval_artifact_path=noisy_eval_artifact_path,
        audit_manifests=audit_manifests,
        experiment_id=experiment_id,
        config=config,
    )
    return {
        "catalog": catalog,
        "split_registry": split_registry,
        "validation_manifest": validation_manifest,
        "test_manifest": test_manifest,
        "audit_manifests": audit_manifests,
        "noisy_eval_artifact": None
        if noisy_eval_artifact_path is None
        else load_noisy_eval_recipe(noisy_eval_artifact_path),
    }
