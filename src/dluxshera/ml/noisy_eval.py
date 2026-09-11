from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from dluxshera.datasets.schema import json_ready, read_json

from .catalog import SampleCatalog
from .noise import noise_config_identity, pair_noise_side_seeds
from .pairs import PairManifest, pair_manifest_content_hash
from .splits import SplitRegistry, split_registry_content_sha256

__all__ = [
    "NOISY_EVAL_RECIPE_SCHEMA_VERSION",
    "load_noisy_eval_recipe",
    "noisy_eval_recipe_content_sha256",
    "validate_noisy_eval_recipe",
]

NOISY_EVAL_RECIPE_SCHEMA_VERSION = "dluxshera_ml_noisy_eval_recipe/1"


def noisy_eval_recipe_content_sha256(recipe: Mapping[str, Any]) -> str:
    stable = dict(recipe)
    stable.pop("generated_at", None)
    stable.pop("content_identity", None)
    raw = json.dumps(json_ready(stable), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_noisy_eval_recipe(path: Path) -> dict[str, Any]:
    payload = read_json(Path(path))
    if payload.get("schema_version") != NOISY_EVAL_RECIPE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported noisy-eval recipe schema {payload.get('schema_version')!r}.")
    identity = payload.get("content_identity", {})
    if isinstance(identity, Mapping) and identity.get("sha256"):
        actual = noisy_eval_recipe_content_sha256(payload)
        if str(identity["sha256"]) != actual:
            raise ValueError(
                "Noisy-eval recipe content_identity.sha256 does not match content "
                f"({identity['sha256']} != {actual})."
            )
    return dict(payload)


def _manifest_hash(pair_manifest: PairManifest) -> str:
    return pair_manifest_content_hash(pair_manifest.manifest, pair_manifest.records)


def validate_noisy_eval_recipe(
    recipe: Mapping[str, Any],
    *,
    study: Mapping[str, Any],
    config: Mapping[str, Any],
    catalog: SampleCatalog,
    split_registry: SplitRegistry,
    validation_manifest: PairManifest,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate a frozen S12 noisy-validation recipe against runtime artifacts."""
    if recipe.get("schema_version") != NOISY_EVAL_RECIPE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported noisy-eval recipe schema {recipe.get('schema_version')!r}.")
    expected_schema = expected.get("schema_version")
    if expected_schema and str(expected_schema) != NOISY_EVAL_RECIPE_SCHEMA_VERSION:
        raise ValueError(
            "Noisy-eval study declaration schema_version does not match runtime recipe schema "
            f"({expected_schema} != {NOISY_EVAL_RECIPE_SCHEMA_VERSION})."
        )
    expected_artifact_id = expected.get("artifact_id")
    if expected_artifact_id and str(recipe.get("artifact_id")) != str(expected_artifact_id):
        raise ValueError(
            "Noisy-eval artifact_id does not match study declaration "
            f"({recipe.get('artifact_id')} != {expected_artifact_id})."
        )
    actual_hash = noisy_eval_recipe_content_sha256(recipe)
    identity = recipe.get("content_identity", {})
    if isinstance(identity, Mapping) and identity.get("sha256") and str(identity["sha256"]) != actual_hash:
        raise ValueError(
            "Noisy-eval recipe content_identity.sha256 does not match content "
            f"({identity['sha256']} != {actual_hash})."
        )
    expected_hash = expected.get("content_sha256")
    if expected_hash and str(expected_hash) != actual_hash:
        raise ValueError(
            "Noisy-eval recipe content_sha256 does not match frozen study declaration "
            f"({expected_hash} != {actual_hash})."
        )
    if str(recipe.get("study_id")) != str(study.get("study_id")):
        raise ValueError(
            f"Noisy-eval study_id {recipe.get('study_id')!r} does not match {study.get('study_id')!r}."
        )
    validation_key = str(config.get("validation_artifact", "validation"))
    artifact_key = recipe.get("artifact_key")
    if artifact_key not in (None, "", validation_key):
        raise ValueError(
            f"Noisy-eval artifact_key {artifact_key!r} does not match validation artifact {validation_key!r}."
        )
    prepared = recipe.get("prepared_dataset", {})
    if prepared.get("artifact_id") != catalog.artifact_id:
        raise ValueError(
            "Noisy-eval prepared artifact_id does not match runtime catalog "
            f"({prepared.get('artifact_id')} != {catalog.artifact_id})."
        )
    if prepared.get("prepared_dataset_hash") != catalog.prepared_dataset_hash:
        raise ValueError(
            "Noisy-eval prepared_dataset_hash does not match runtime catalog "
            f"({prepared.get('prepared_dataset_hash')} != {catalog.prepared_dataset_hash})."
        )
    split = recipe.get("split_registry", {})
    split_hash = split_registry_content_sha256(split_registry)
    if split.get("artifact_id") != split_registry.artifact_id:
        raise ValueError(
            "Noisy-eval split registry artifact_id does not match runtime split "
            f"({split.get('artifact_id')} != {split_registry.artifact_id})."
        )
    if split.get("content_sha256") != split_hash:
        raise ValueError(
            "Noisy-eval split registry hash does not match runtime split "
            f"({split.get('content_sha256')} != {split_hash})."
        )
    pair = recipe.get("underlying_pair_manifest", {})
    manifest_hash = _manifest_hash(validation_manifest)
    if pair.get("artifact_id") != validation_manifest.artifact_id:
        raise ValueError(
            "Noisy-eval validation pair manifest artifact_id does not match runtime manifest "
            f"({pair.get('artifact_id')} != {validation_manifest.artifact_id})."
        )
    if pair.get("content_sha256") != manifest_hash:
        raise ValueError(
            "Noisy-eval validation pair manifest hash does not match runtime manifest "
            f"({pair.get('content_sha256')} != {manifest_hash})."
        )
    noise_identity = noise_config_identity(config.get("validation_noise"))
    if recipe.get("noise_model") != noise_identity:
        raise ValueError(
            "Noisy-eval validation noise identity does not match run configuration."
        )
    records = recipe.get("records")
    if not isinstance(records, list):
        raise ValueError("Noisy-eval recipe must contain a records list.")
    expected_count = expected.get("record_count", expected.get("ordered_pair_count"))
    if expected_count is not None and int(expected_count) != len(records):
        raise ValueError(
            "Noisy-eval record count does not match study declaration "
            f"({expected_count} != {len(records)})."
        )
    if len(records) != len(validation_manifest.records):
        raise ValueError(
            "Noisy-eval record count does not match validation manifest "
            f"({len(records)} != {len(validation_manifest.records)})."
        )
    for index, (frozen, record) in enumerate(zip(records, validation_manifest.records)):
        if int(frozen.get("index", -1)) != index:
            raise ValueError(f"Noisy-eval record {index} has mismatched index {frozen.get('index')!r}.")
        for field, value in {
            "pair_record_id": record.pair_record_id,
            "sample_a_id": record.sample_a_id,
            "sample_b_id": record.sample_b_id,
        }.items():
            if str(frozen.get(field)) != str(value):
                raise ValueError(f"Noisy-eval record {index} {field} does not match validation manifest.")
        expected_seeds = pair_noise_side_seeds(
            noise_identity,
            pair_record_id=record.pair_record_id,
            dynamic_seed_offset=index,
        )
        if dict(frozen.get("seeds", {})) != expected_seeds:
            raise ValueError(f"Noisy-eval record {index} seed schedule does not match runtime dataset.")
    return {
        "artifact_id": recipe.get("artifact_id"),
        "content_sha256": actual_hash,
        "record_count": len(records),
        "validation_artifact": validation_manifest.artifact_id,
        "validation_artifact_hash": manifest_hash,
        "noise_model": noise_identity,
    }
