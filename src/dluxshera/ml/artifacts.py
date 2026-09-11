from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dluxshera.datasets.schema import json_ready, read_json, write_json

from .catalog import SampleCatalog
from .pairs import PairManifest, pair_manifest_content_hash
from .scaling import IntensityScaler, intensity_scaler_content_sha256
from .splits import SplitRegistry, split_registry_content_sha256

__all__ = [
    "ArtifactLock",
    "artifact_lock_content_sha256",
    "build_artifact_lock",
    "load_artifact_lock",
    "validate_artifact_lock",
    "write_artifact_lock",
]

ARTIFACT_LOCK_SCHEMA_VERSION = "dluxshera_ml_artifact_lock/1"


@dataclass(frozen=True)
class ArtifactLock:
    """Record portable scientific identities for cluster-produced artifacts."""

    lock_id: str
    study_id: str
    schema_version: str
    prepared_dataset: Mapping[str, Any]
    split_registry: Mapping[str, Any]
    scaler: Mapping[str, Any]
    pair_manifests: Mapping[str, Mapping[str, Any]]
    auxiliary_artifacts: Mapping[str, Mapping[str, Any]]
    nuisance_assignment: Mapping[str, Any]
    source_identity: Mapping[str, Any]
    recipe_identities: Mapping[str, Any]
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable lock payload."""
        return {
            "schema_version": self.schema_version,
            "lock_id": self.lock_id,
            "study_id": self.study_id,
            "prepared_dataset": dict(self.prepared_dataset),
            "split_registry": dict(self.split_registry),
            "scaler": dict(self.scaler),
            "pair_manifests": {
                str(key): dict(value) for key, value in self.pair_manifests.items()
            },
            "auxiliary_artifacts": {
                str(key): dict(value) for key, value in self.auxiliary_artifacts.items()
            },
            "nuisance_assignment": dict(self.nuisance_assignment),
            "source_identity": dict(self.source_identity),
            "recipe_identities": dict(self.recipe_identities),
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactLock":
        """Build a lock from serialized JSON."""
        if payload.get("schema_version") != ARTIFACT_LOCK_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported artifact lock schema {payload.get('schema_version')!r}."
            )
        return cls(
            schema_version=str(payload["schema_version"]),
            lock_id=str(payload["lock_id"]),
            study_id=str(payload["study_id"]),
            prepared_dataset=dict(payload.get("prepared_dataset", {})),
            split_registry=dict(payload.get("split_registry", {})),
            scaler=dict(payload.get("scaler", {})),
            pair_manifests={
                str(key): dict(value)
                for key, value in dict(payload.get("pair_manifests", {})).items()
            },
            auxiliary_artifacts={
                str(key): dict(value)
                for key, value in dict(payload.get("auxiliary_artifacts", {})).items()
            },
            nuisance_assignment=dict(payload.get("nuisance_assignment", {})),
            source_identity=dict(payload.get("source_identity", {})),
            recipe_identities=dict(payload.get("recipe_identities", {})),
            generated_at=str(payload.get("generated_at")),
        )


def artifact_lock_content_sha256(lock: ArtifactLock | Mapping[str, Any]) -> str:
    """Return the stable content hash for a lock file."""
    payload = lock.to_dict() if isinstance(lock, ArtifactLock) else dict(lock)
    stable = dict(payload)
    stable.pop("generated_at", None)
    stable.pop("content_identity", None)
    raw = json.dumps(json_ready(stable), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _pair_identity(pair_manifest: PairManifest) -> dict[str, Any]:
    return {
        "artifact_id": pair_manifest.artifact_id,
        "content_sha256": pair_manifest_content_hash(
            pair_manifest.manifest,
            pair_manifest.records,
        ),
        "recipe_identity": pair_manifest.manifest.get("recipe_identity"),
        "pair_count": len(pair_manifest.records),
    }


def build_artifact_lock(
    *,
    study: Mapping[str, Any],
    catalog: SampleCatalog,
    split_registry: SplitRegistry,
    scaler: IntensityScaler,
    pair_manifests: Mapping[str, PairManifest],
    lock_id: str | None = None,
) -> ArtifactLock:
    """Build a portable artifact lock from materialized study artifacts."""
    study_id = str(study.get("study_id", "UNKNOWN"))
    prepared_source = catalog.manifest.get("source_dataset", {})
    source_identity = {
        "dataset_version": prepared_source.get("dataset_version"),
        "master_scientific_content_hash": prepared_source.get(
            "master_scientific_content_hash"
        ),
        "render_contract_hash": prepared_source.get("render_contract_hash"),
        "render_system_contract_hash": prepared_source.get("render_system_contract_hash"),
        "science_vector_space_id": prepared_source.get("science_vector_space_id"),
        "nuisance_vector_space_id": prepared_source.get("nuisance_vector_space_id"),
        "nuisance_bank_hash": prepared_source.get("nuisance_bank_hash"),
    }
    pair_identities = {
        str(key): _pair_identity(value) for key, value in pair_manifests.items()
    }
    scaler_payload = scaler.to_dict()
    scaler_identity = {
        "content_sha256": intensity_scaler_content_sha256(scaler_payload),
        "mode": scaler.mode,
        "scale": scaler.scale,
        "sample_count": scaler.sample_count,
        "source_population": scaler_payload.get("source_population", {}),
    }
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    provisional = ArtifactLock(
        lock_id=str(lock_id or f"{study_id}-ARTIFACT-LOCK-v1"),
        study_id=study_id,
        schema_version=ARTIFACT_LOCK_SCHEMA_VERSION,
        prepared_dataset={
            "artifact_id": catalog.artifact_id,
            "prepared_dataset_hash": catalog.prepared_dataset_hash,
            "manifest_content_identity": catalog.manifest.get("content_identity"),
            "sample_count": catalog.sample_count,
            "science_dim": catalog.science_dim,
            "vector_space_labels": list(catalog.parameter_labels),
        },
        split_registry={
            "artifact_id": split_registry.artifact_id,
            "content_sha256": split_registry_content_sha256(split_registry),
            "prepared_dataset_hash": split_registry.prepared_dataset.get(
                "prepared_dataset_hash"
            ),
        },
        scaler=scaler_identity,
        pair_manifests=pair_identities,
        auxiliary_artifacts={
            str(key): dict(value)
            for key, value in dict(study.get("auxiliary_artifacts", {}) or {}).items()
        },
        nuisance_assignment={
            "nuisance_group_policy": split_registry.nuisance_group_policy,
            "nuisance_assignments": dict(sorted(split_registry.nuisance_assignments.items())),
            "counts": dict(split_registry.counts.get("nuisance_groups", {})),
        },
        source_identity=source_identity,
        recipe_identities={
            "evaluation_artifacts": {
                str(key): {
                    item_key: item_value
                    for item_key, item_value in dict(value).items()
                    if item_key in {
                        "artifact_id",
                        "pair_policy_id",
                        "split",
                        "seed",
                        "pairs_per_slice",
                        "ordered_pair_count",
                        "content_sha256",
                    }
                }
                for key, value in dict(study.get("evaluation_artifacts", {})).items()
            }
        },
        generated_at=generated_at,
    )
    payload = provisional.to_dict()
    payload["content_identity"] = {
        "algorithm": "sha256/json-canonical/artifact-lock-v1",
        "sha256": artifact_lock_content_sha256(provisional),
        "excludes": ["generated_at", "content_identity"],
    }
    return ArtifactLock.from_dict(payload)


def write_artifact_lock(path: Path, lock: ArtifactLock, *, overwrite: bool = False) -> None:
    """Write an artifact lock JSON file."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it.")
    payload = lock.to_dict()
    payload["content_identity"] = {
        "algorithm": "sha256/json-canonical/artifact-lock-v1",
        "sha256": artifact_lock_content_sha256(lock),
        "excludes": ["generated_at", "content_identity"],
    }
    write_json(path, payload)


def load_artifact_lock(path: Path) -> ArtifactLock:
    """Load and validate a portable artifact lock."""
    payload = read_json(Path(path))
    lock = ArtifactLock.from_dict(payload)
    identity = payload.get("content_identity", {})
    if isinstance(identity, Mapping) and identity.get("sha256"):
        actual = artifact_lock_content_sha256(lock)
        if str(identity["sha256"]) != actual:
            raise ValueError(
                "Artifact lock content_identity.sha256 does not match lock content "
                f"({identity['sha256']} != {actual})."
            )
    return lock


def validate_artifact_lock(
    *,
    lock: ArtifactLock,
    study: Mapping[str, Any],
    catalog: SampleCatalog,
    split_registry: SplitRegistry,
    scaler: IntensityScaler | None = None,
    pair_manifests: Mapping[str, PairManifest] | None = None,
) -> None:
    """Validate materialized artifacts against a portable lock."""
    expected_study = str(study.get("study_id", ""))
    if expected_study and lock.study_id != expected_study:
        raise ValueError(
            f"Artifact lock study_id {lock.study_id!r} does not match study {expected_study!r}."
        )
    prepared = lock.prepared_dataset
    if prepared.get("artifact_id") != catalog.artifact_id:
        raise ValueError(
            "Artifact lock prepared dataset artifact_id does not match current catalog "
            f"({prepared.get('artifact_id')} != {catalog.artifact_id})."
        )
    if prepared.get("prepared_dataset_hash") != catalog.prepared_dataset_hash:
        raise ValueError(
            "Artifact lock prepared dataset hash does not match current catalog "
            f"({prepared.get('prepared_dataset_hash')} != {catalog.prepared_dataset_hash})."
        )
    split = lock.split_registry
    actual_split_hash = split_registry_content_sha256(split_registry)
    if split.get("artifact_id") != split_registry.artifact_id:
        raise ValueError(
            "Artifact lock split registry artifact_id does not match current registry "
            f"({split.get('artifact_id')} != {split_registry.artifact_id})."
        )
    if split.get("content_sha256") != actual_split_hash:
        raise ValueError(
            "Artifact lock split registry hash does not match current registry "
            f"({split.get('content_sha256')} != {actual_split_hash})."
        )
    if lock.scaler.get("content_sha256") and scaler is None:
        raise ValueError("Artifact lock records a scaler; supply the scaler for validation.")
    if scaler is not None:
        actual_scaler_hash = intensity_scaler_content_sha256(scaler.to_dict())
        if lock.scaler.get("content_sha256") != actual_scaler_hash:
            raise ValueError(
                "Artifact lock scaler hash does not match current scaler "
                f"({lock.scaler.get('content_sha256')} != {actual_scaler_hash})."
            )
    for key, manifest in dict(pair_manifests or {}).items():
        expected = lock.pair_manifests.get(str(key))
        if expected is None:
            raise ValueError(f"Artifact lock has no pair manifest entry for {key!r}.")
        actual = _pair_identity(manifest)
        for field in ("artifact_id", "content_sha256"):
            if expected.get(field) != actual.get(field):
                raise ValueError(
                    f"Artifact lock pair manifest {key!r} {field} does not match "
                    f"({expected.get(field)} != {actual.get(field)})."
                )
