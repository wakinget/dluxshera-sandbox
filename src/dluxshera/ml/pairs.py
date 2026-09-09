from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dluxshera.datasets.schema import read_json, read_jsonl, write_json, write_jsonl

from .catalog import SampleCatalog
from .splits import SplitRegistry, split_registry_content_sha256

__all__ = [
    "DEFAULT_FISHER_DISTANCE_BIN_EDGES",
    "PairManifest",
    "PairPolicy",
    "PairRecord",
    "PairSampler",
    "fisher_distance_bin_label",
    "generate_frozen_pair_manifest",
    "load_pair_manifest",
    "make_reverse_pair_record",
    "pair_manifest_content_hash",
    "write_pair_manifest",
]

PAIR_MANIFEST_SCHEMA_VERSION = "dluxshera_ml_pair_eval_manifest/1"
PAIR_POLICY_SCHEMA_VERSION = "dluxshera_ml_pair_policy/1"
DEFAULT_PAIR_ARTIFACT_ID = "PAIR-EVAL-v1"
DEFAULT_FISHER_DISTANCE_BIN_EDGES = (0.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0)

PAIR_FAMILIES = {
    "same_nuisance_different_science",
    "same_science_different_nuisance",
    "different_nuisance_same_science",
    "different_science_different_nuisance",
    "identity",
}

PAIR_FAMILY_ALIASES = {
    "A": "same_nuisance_different_science",
    "B": "different_nuisance_same_science",
    "C": "different_science_different_nuisance",
    "I": "identity",
}


def _stable_id(parts: Sequence[Any], *, prefix: str = "pair") -> str:
    raw = json.dumps(list(parts), sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


def _stable_sha256(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _counts(values: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return dict(sorted(out.items()))


def _distance_summary(distances: Sequence[float]) -> dict[str, float | None]:
    if not distances:
        return {"min": None, "max": None, "mean": None, "p50": None, "p90": None}
    values = np.asarray(distances, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def _canonical_family(name: str) -> str:
    family = PAIR_FAMILY_ALIASES.get(str(name), str(name))
    if family == "same_science_different_nuisance":
        return "different_nuisance_same_science"
    return family


def _normalize_weights(
    weights: Mapping[str, float] | None,
    *,
    aliases: Mapping[str, str] | None = None,
    name: str,
) -> dict[str, float]:
    if not weights:
        return {}
    aliases = {} if aliases is None else aliases
    out: dict[str, float] = {}
    for key, value in weights.items():
        label = aliases.get(str(key), str(key))
        weight = float(value)
        if weight < 0.0 or not np.isfinite(weight):
            raise ValueError(f"{name} weights must be finite and >= 0.")
        out[label] = out.get(label, 0.0) + weight
    return out


def _choose_weighted_key(rng: np.random.Generator, weights: Mapping[str, float]) -> str:
    keys = list(weights)
    raw = np.asarray([float(weights[key]) for key in keys], dtype=np.float64)
    if raw.size == 0 or float(np.sum(raw)) <= 0.0:
        raise ValueError("At least one selection weight must be positive.")
    return str(rng.choice(keys, p=raw / np.sum(raw)))


def _deep_update(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if (
            key not in {"family_weights", "dataset_family_weights", "distance_bin_weights"}
            and isinstance(value, Mapping)
            and isinstance(out.get(key), Mapping)
        ):
            out[key] = _deep_update(dict(out[key]), value)
        else:
            out[key] = value
    return out


def fisher_distance_bin_label(distance: float, edges: Sequence[float]) -> str:
    """Return the configured Fisher-distance bin label for ``distance``.

    Bins are left-closed/right-open except for the final bin, which includes
    exactly the upper edge. Values outside the configured range are reported as
    ``below_range`` or ``above_range``.
    """
    values = tuple(float(v) for v in edges)
    if len(values) < 2:
        raise ValueError("Fisher-distance edges must contain at least two values.")
    if any(hi <= lo for lo, hi in zip(values[:-1], values[1:])):
        raise ValueError("Fisher-distance edges must be strictly increasing.")

    def fmt(value: float) -> str:
        return str(int(value)) if float(value).is_integer() else f"{value:g}"

    d = float(distance)
    if d < values[0]:
        return "below_range"
    if d > values[-1]:
        return "above_range"
    for index, (lo, hi) in enumerate(zip(values[:-1], values[1:])):
        if index == len(values) - 2:
            if lo <= d <= hi:
                return f"{fmt(lo)}-{fmt(hi)}"
        elif lo <= d < hi:
            return f"{fmt(lo)}-{fmt(hi)}"
    return "above_range"


@dataclass(frozen=True)
class PairPolicy:
    """Describe an ordered-pair sampling policy without materializing pairs."""

    policy_id: str = "generic_pair_policy_v1"
    schema_version: str = PAIR_POLICY_SCHEMA_VERSION
    family_weights: Mapping[str, float] = field(
        default_factory=lambda: {"same_nuisance_different_science": 1.0}
    )
    same_pair_id: bool = False
    min_fisher_distance: float = 0.0
    max_fisher_distance: float | None = None
    max_changed_science_dimensions: int | None = None
    allow_identity_pairs: bool = False
    include_reverse: bool = False
    dataset_families: tuple[str, ...] = ()
    dataset_family_weights: Mapping[str, float] = field(default_factory=dict)
    dataset_family_sampling: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    require_same_dataset_family: bool = False
    joint_train_prefix_size: int | None = None
    fisher_distance_bin_edges: tuple[float, ...] = DEFAULT_FISHER_DISTANCE_BIN_EDGES
    fisher_distance_bins: tuple[str, ...] = ()
    distance_bin_weights: Mapping[str, float] = field(default_factory=dict)
    curriculum_stages: tuple[Mapping[str, Any], ...] = ()
    max_sampling_attempts: int = 1000

    def __post_init__(self) -> None:
        family_weights = _normalize_weights(
            self.family_weights,
            aliases=PAIR_FAMILY_ALIASES,
            name="Pair-family",
        )
        if family_weights != dict(self.family_weights):
            object.__setattr__(self, "family_weights", family_weights)
        for family, weight in self.family_weights.items():
            if family not in PAIR_FAMILIES:
                raise ValueError(f"Unsupported pair family {family!r}.")
            if float(weight) < 0.0 or not np.isfinite(float(weight)):
                raise ValueError("Pair-family weights must be finite and >= 0.")
        if sum(float(v) for v in self.family_weights.values()) <= 0.0:
            raise ValueError("At least one pair-family weight must be positive.")
        dataset_weights = _normalize_weights(
            self.dataset_family_weights,
            name="Dataset-family",
        )
        if dataset_weights != dict(self.dataset_family_weights):
            object.__setattr__(self, "dataset_family_weights", dataset_weights)
        if dataset_weights and sum(float(v) for v in dataset_weights.values()) <= 0.0:
            raise ValueError("At least one Dataset-family weight must be positive.")
        family_sampling: dict[str, dict[str, Any]] = {}
        for family_name, raw_block in dict(self.dataset_family_sampling or {}).items():
            if not isinstance(raw_block, Mapping):
                raise ValueError("dataset_family_sampling entries must be mappings.")
            block = dict(raw_block)
            weight = float(block.get("weight", 1.0))
            if weight < 0.0 or not np.isfinite(weight):
                raise ValueError("dataset_family_sampling weights must be finite and >= 0.")
            block["weight"] = weight
            family_sampling[str(family_name)] = block
        if family_sampling and sum(float(v["weight"]) for v in family_sampling.values()) <= 0.0:
            raise ValueError("At least one dataset_family_sampling weight must be positive.")
        object.__setattr__(self, "dataset_family_sampling", family_sampling)
        if float(self.min_fisher_distance) < 0.0:
            raise ValueError("min_fisher_distance must be >= 0.")
        if self.max_fisher_distance is not None and float(self.max_fisher_distance) < float(
            self.min_fisher_distance
        ):
            raise ValueError("max_fisher_distance must be >= min_fisher_distance.")
        edges = tuple(float(v) for v in self.fisher_distance_bin_edges)
        if len(edges) < 2 or any(hi <= lo for lo, hi in zip(edges[:-1], edges[1:])):
            raise ValueError("fisher_distance_bin_edges must be strictly increasing.")
        object.__setattr__(self, "fisher_distance_bin_edges", edges)
        if self.joint_train_prefix_size is not None and int(self.joint_train_prefix_size) < 1:
            raise ValueError("joint_train_prefix_size must be >= 1 when provided.")
        if self.fisher_distance_bins:
            valid_bins = {
                fisher_distance_bin_label((lo + hi) / 2.0, edges)
                for lo, hi in zip(edges[:-1], edges[1:])
            }
            requested = {str(label) for label in self.fisher_distance_bins}
            unknown = sorted(requested - valid_bins)
            if unknown:
                raise ValueError(f"Unknown fisher distance bin label(s): {unknown}.")
            object.__setattr__(
                self,
                "fisher_distance_bins",
                tuple(str(label) for label in self.fisher_distance_bins),
            )
        distance_weights = _normalize_weights(
            self.distance_bin_weights,
            name="Distance-bin",
        )
        for family_name, block in self.dataset_family_sampling.items():
            block_weights = _normalize_weights(
                block.get("distance_bin_weights", {}),
                name=f"Distance-bin for {family_name}",
            )
            block_bins = tuple(str(v) for v in block.get("fisher_distance_bins", []) or [])
            valid_bins = {
                fisher_distance_bin_label((lo + hi) / 2.0, edges)
                for lo, hi in zip(edges[:-1], edges[1:])
            }
            unknown_weights = sorted(set(block_weights) - valid_bins)
            unknown_bins = sorted(set(block_bins) - valid_bins)
            if unknown_weights:
                raise ValueError(
                    f"Unknown dataset_family_sampling.{family_name}.distance_bin_weights "
                    f"label(s): {unknown_weights}."
                )
            if unknown_bins:
                raise ValueError(
                    f"Unknown dataset_family_sampling.{family_name}.fisher_distance_bins "
                    f"label(s): {unknown_bins}."
                )
            if block_weights and sum(float(v) for v in block_weights.values()) <= 0.0:
                raise ValueError(
                    f"At least one distance-bin weight must be positive for {family_name}."
                )
            block["distance_bin_weights"] = block_weights
            if block_bins:
                block["fisher_distance_bins"] = block_bins
        if distance_weights:
            valid_bins = {
                fisher_distance_bin_label((lo + hi) / 2.0, edges)
                for lo, hi in zip(edges[:-1], edges[1:])
            }
            unknown = sorted(set(distance_weights) - valid_bins)
            if unknown:
                raise ValueError(f"Unknown distance_bin_weights label(s): {unknown}.")
        if distance_weights and sum(float(v) for v in distance_weights.values()) <= 0.0:
            raise ValueError("At least one distance-bin weight must be positive.")
        object.__setattr__(self, "distance_bin_weights", distance_weights)
        stages = tuple(dict(stage) for stage in self.curriculum_stages)
        previous_epoch = -1
        for stage in stages:
            start_epoch = int(stage.get("start_epoch", 0))
            if start_epoch < 0:
                raise ValueError("curriculum stage start_epoch must be >= 0.")
            if start_epoch <= previous_epoch:
                raise ValueError("curriculum stages must be ordered by increasing start_epoch.")
            previous_epoch = start_epoch
        object.__setattr__(self, "curriculum_stages", stages)
        if self.max_changed_science_dimensions is not None and int(
            self.max_changed_science_dimensions
        ) < 1:
            raise ValueError("max_changed_science_dimensions must be >= 1 when provided.")
        if int(self.max_sampling_attempts) < 1:
            raise ValueError("max_sampling_attempts must be >= 1.")

    def to_dict(self) -> dict[str, Any]:
        """Return policy metadata suitable for manifests and run configs."""
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "family_weights": dict(self.family_weights),
            "same_pair_id": bool(self.same_pair_id),
            "min_fisher_distance": float(self.min_fisher_distance),
            "max_fisher_distance": None
            if self.max_fisher_distance is None
            else float(self.max_fisher_distance),
            "max_changed_science_dimensions": self.max_changed_science_dimensions,
            "allow_identity_pairs": bool(self.allow_identity_pairs),
            "include_reverse": bool(self.include_reverse),
            "dataset_families": list(self.dataset_families),
            "dataset_family_weights": dict(self.dataset_family_weights),
            "dataset_family_sampling": {
                str(key): dict(value)
                for key, value in self.dataset_family_sampling.items()
            },
            "require_same_dataset_family": bool(self.require_same_dataset_family),
            "joint_train_prefix_size": self.joint_train_prefix_size,
            "fisher_distance_bin_edges": [float(v) for v in self.fisher_distance_bin_edges],
            "fisher_distance_bins": list(self.fisher_distance_bins),
            "distance_bin_weights": dict(self.distance_bin_weights),
            "curriculum_stages": [dict(stage) for stage in self.curriculum_stages],
            "max_sampling_attempts": int(self.max_sampling_attempts),
            "target_convention": "target_delta_z = z_B - z_A",
            "ordered_pair_semantics": "A=current/reference/model, B=target/observation",
            "pair_family_vocabulary": {
                "A": "same_nuisance_different_science",
                "B": "different_nuisance_same_science",
                "C": "different_science_different_nuisance",
                "I": "identity",
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PairPolicy":
        """Build a pair policy from serialized metadata."""
        defaults = cls()
        raw_max_fisher_distance = payload.get(
            "max_fisher_distance",
            defaults.max_fisher_distance,
        )
        return cls(
            policy_id=str(payload.get("policy_id", defaults.policy_id)),
            schema_version=str(payload.get("schema_version", defaults.schema_version)),
            family_weights=dict(
                payload.get("family_weights", defaults.family_weights)
            ),
            same_pair_id=bool(payload.get("same_pair_id", defaults.same_pair_id)),
            min_fisher_distance=float(
                payload.get("min_fisher_distance", defaults.min_fisher_distance)
            ),
            max_fisher_distance=None
            if raw_max_fisher_distance is None
            else float(raw_max_fisher_distance),
            max_changed_science_dimensions=payload.get(
                "max_changed_science_dimensions",
                defaults.max_changed_science_dimensions,
            ),
            allow_identity_pairs=bool(
                payload.get("allow_identity_pairs", defaults.allow_identity_pairs)
            ),
            include_reverse=bool(payload.get("include_reverse", defaults.include_reverse)),
            dataset_families=tuple(str(v) for v in payload.get("dataset_families", []) or []),
            dataset_family_weights=dict(payload.get("dataset_family_weights", {}) or {}),
            dataset_family_sampling={
                str(key): dict(value)
                for key, value in dict(payload.get("dataset_family_sampling", {}) or {}).items()
            },
            require_same_dataset_family=bool(
                payload.get("require_same_dataset_family", defaults.require_same_dataset_family)
            ),
            joint_train_prefix_size=None
            if payload.get("joint_train_prefix_size") is None
            else int(payload.get("joint_train_prefix_size")),
            fisher_distance_bin_edges=tuple(
                float(v)
                for v in payload.get(
                    "fisher_distance_bin_edges",
                    defaults.fisher_distance_bin_edges,
                )
            ),
            fisher_distance_bins=tuple(
                str(v) for v in payload.get("fisher_distance_bins", []) or []
            ),
            distance_bin_weights=dict(payload.get("distance_bin_weights", {}) or {}),
            curriculum_stages=tuple(
                dict(stage) for stage in payload.get("curriculum_stages", []) or []
            ),
            max_sampling_attempts=int(
                payload.get("max_sampling_attempts", defaults.max_sampling_attempts)
            ),
        )

    def resolved_for_epoch(self, epoch: int | None) -> "PairPolicy":
        """Return a copy with the latest matching curriculum stage applied."""
        if epoch is None or not self.curriculum_stages:
            return self
        updates: dict[str, Any] = {}
        for stage in self.curriculum_stages:
            if int(epoch) >= int(stage.get("start_epoch", 0)):
                stage_updates = dict(stage)
                stage_updates.pop("start_epoch", None)
                updates = _deep_update(updates, stage_updates)
        if not updates:
            return self
        payload = self.to_dict()
        payload.pop("target_convention", None)
        payload.pop("ordered_pair_semantics", None)
        payload.pop("pair_family_vocabulary", None)
        payload = _deep_update(payload, updates)
        return PairPolicy.from_dict(payload)

    def dataset_family_sampling_weights(self, epoch: int | None = None) -> dict[str, float]:
        """Return effective source-family weights after curriculum resolution."""
        policy = self.resolved_for_epoch(epoch) if epoch is not None else self
        if policy.dataset_family_sampling:
            return {
                str(key): float(value.get("weight", 1.0))
                for key, value in policy.dataset_family_sampling.items()
            }
        if policy.dataset_family_weights:
            return {str(key): float(value) for key, value in policy.dataset_family_weights.items()}
        if policy.dataset_families:
            return {str(key): 1.0 for key in policy.dataset_families}
        return {}

    def resolved_dataset_family_constraints(
        self,
        epoch: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return inspectable effective constraints for each dataset family."""
        policy = self.resolved_for_epoch(epoch) if epoch is not None else self
        weights = policy.dataset_family_sampling_weights()
        out: dict[str, dict[str, Any]] = {}
        for family, weight in weights.items():
            family_policy = policy.resolved_for_dataset_family(family)
            out[str(family)] = {
                "weight": float(weight),
                "min_fisher_distance": float(family_policy.min_fisher_distance),
                "max_fisher_distance": family_policy.max_fisher_distance,
                "fisher_distance_bins": list(family_policy.fisher_distance_bins),
                "distance_bin_weights": dict(family_policy.distance_bin_weights),
                "max_changed_science_dimensions": family_policy.max_changed_science_dimensions,
            }
        return out

    def resolved_for_dataset_family(
        self,
        dataset_family: str | None,
        *,
        epoch: int | None = None,
    ) -> "PairPolicy":
        """Return effective constraints for one selected dataset family."""
        policy = self.resolved_for_epoch(epoch)
        if dataset_family is None or not policy.dataset_family_sampling:
            return policy
        block = dict(policy.dataset_family_sampling.get(str(dataset_family), {}))
        payload = policy.to_dict()
        for key in (
            "weight",
            "target_convention",
            "ordered_pair_semantics",
            "pair_family_vocabulary",
        ):
            payload.pop(key, None)
        payload["dataset_families"] = [str(dataset_family)]
        payload["dataset_family_weights"] = {}
        payload["dataset_family_sampling"] = {}
        for key in (
            "min_fisher_distance",
            "max_fisher_distance",
            "fisher_distance_bins",
            "distance_bin_weights",
            "max_changed_science_dimensions",
        ):
            if key in block:
                payload[key] = block[key]
        if "distance_bin_weights" not in block and "fisher_distance_bins" not in block:
            payload["distance_bin_weights"] = {}
            payload["fisher_distance_bins"] = []
        return PairPolicy.from_dict(payload)


@dataclass(frozen=True)
class PairRecord:
    """Represent one ordered image pair by prepared sample references."""

    pair_record_id: str
    sample_a_id: str
    sample_b_id: str
    sample_a_index: int
    sample_b_index: int
    target_delta_z: tuple[float, ...]
    target_delta_theta: tuple[float, ...]
    nuisance_delta: tuple[float, ...]
    nuisance_a_id: str
    nuisance_b_id: str
    science_a_id: str
    science_b_id: str
    pair_family: str
    split: str
    eval_slice: str | None
    fisher_distance_l2: float
    changed_science_dimensions: int
    dataset_family_a: str
    dataset_family_b: str
    pair_id_a: str
    pair_id_b: str
    prepared_dataset_hash: str
    split_registry_id: str
    pair_policy_id: str
    distance_bin_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable ordered-pair record."""
        return {
            "pair_record_id": self.pair_record_id,
            "sample_a_id": self.sample_a_id,
            "sample_b_id": self.sample_b_id,
            "sample_a_index": int(self.sample_a_index),
            "sample_b_index": int(self.sample_b_index),
            "target_delta_z": list(self.target_delta_z),
            "target_delta_theta": list(self.target_delta_theta),
            "nuisance_delta": list(self.nuisance_delta),
            "nuisance_a_id": self.nuisance_a_id,
            "nuisance_b_id": self.nuisance_b_id,
            "science_a_id": self.science_a_id,
            "science_b_id": self.science_b_id,
            "pair_family": self.pair_family,
            "split": self.split,
            "eval_slice": self.eval_slice,
            "fisher_distance_l2": float(self.fisher_distance_l2),
            "changed_science_dimensions": int(self.changed_science_dimensions),
            "dataset_family_a": self.dataset_family_a,
            "dataset_family_b": self.dataset_family_b,
            "pair_id_a": self.pair_id_a,
            "pair_id_b": self.pair_id_b,
            "prepared_dataset_hash": self.prepared_dataset_hash,
            "split_registry_id": self.split_registry_id,
            "pair_policy_id": self.pair_policy_id,
            "distance_bin_label": self.distance_bin_label,
            "target_convention": "B_minus_A",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PairRecord":
        """Build a pair record from a manifest row."""
        return cls(
            pair_record_id=str(payload["pair_record_id"]),
            sample_a_id=str(payload["sample_a_id"]),
            sample_b_id=str(payload["sample_b_id"]),
            sample_a_index=int(payload["sample_a_index"]),
            sample_b_index=int(payload["sample_b_index"]),
            target_delta_z=tuple(float(v) for v in payload["target_delta_z"]),
            target_delta_theta=tuple(float(v) for v in payload["target_delta_theta"]),
            nuisance_delta=tuple(float(v) for v in payload.get("nuisance_delta", [])),
            nuisance_a_id=str(payload["nuisance_a_id"]),
            nuisance_b_id=str(payload["nuisance_b_id"]),
            science_a_id=str(payload["science_a_id"]),
            science_b_id=str(payload["science_b_id"]),
            pair_family=str(payload["pair_family"]),
            split=str(payload["split"]),
            eval_slice=None if payload.get("eval_slice") is None else str(payload["eval_slice"]),
            fisher_distance_l2=float(payload["fisher_distance_l2"]),
            changed_science_dimensions=int(payload["changed_science_dimensions"]),
            dataset_family_a=str(payload.get("dataset_family_a", "")),
            dataset_family_b=str(payload.get("dataset_family_b", "")),
            pair_id_a=str(payload.get("pair_id_a", "")),
            pair_id_b=str(payload.get("pair_id_b", "")),
            prepared_dataset_hash=str(payload["prepared_dataset_hash"]),
            split_registry_id=str(payload["split_registry_id"]),
            pair_policy_id=str(payload["pair_policy_id"]),
            distance_bin_label=(
                None
                if payload.get("distance_bin_label") is None
                else str(payload.get("distance_bin_label"))
            ),
        )


def make_reverse_pair_record(
    record: PairRecord,
    *,
    pair_record_id: str | None = None,
    id_prefix: str = "pair",
) -> PairRecord:
    """Return the ordered reverse of ``record`` with antisymmetric targets."""
    if pair_record_id is None:
        pair_record_id = _stable_id(
            [record.pair_record_id, "reverse", record.sample_b_id, record.sample_a_id],
            prefix=id_prefix,
        )
    return PairRecord(
        pair_record_id=str(pair_record_id),
        sample_a_id=record.sample_b_id,
        sample_b_id=record.sample_a_id,
        sample_a_index=int(record.sample_b_index),
        sample_b_index=int(record.sample_a_index),
        target_delta_z=tuple(-float(v) for v in record.target_delta_z),
        target_delta_theta=tuple(-float(v) for v in record.target_delta_theta),
        nuisance_delta=tuple(-float(v) for v in record.nuisance_delta),
        nuisance_a_id=record.nuisance_b_id,
        nuisance_b_id=record.nuisance_a_id,
        science_a_id=record.science_b_id,
        science_b_id=record.science_a_id,
        pair_family=record.pair_family,
        split=record.split,
        eval_slice=record.eval_slice,
        fisher_distance_l2=float(record.fisher_distance_l2),
        changed_science_dimensions=int(record.changed_science_dimensions),
        dataset_family_a=record.dataset_family_b,
        dataset_family_b=record.dataset_family_a,
        pair_id_a=record.pair_id_b,
        pair_id_b=record.pair_id_a,
        prepared_dataset_hash=record.prepared_dataset_hash,
        split_registry_id=record.split_registry_id,
        pair_policy_id=record.pair_policy_id,
        distance_bin_label=record.distance_bin_label,
    )


@dataclass(frozen=True)
class PairManifest:
    """Hold frozen validation/test pair records plus provenance metadata."""

    artifact_id: str
    manifest: Mapping[str, Any]
    records: tuple[PairRecord, ...]

    def summary(self) -> dict[str, Any]:
        """Return compact pair-manifest counts and distance summaries."""
        return {
            "artifact_id": self.artifact_id,
            "content_sha256": self.manifest.get("content_identity", {}).get("sha256"),
            "pair_count": len(self.records),
            "pair_family_counts": _counts([record.pair_family for record in self.records]),
            "eval_slice_counts": _counts(
                [record.eval_slice or "none" for record in self.records]
            ),
            "distance_summary": _distance_summary(
                [record.fisher_distance_l2 for record in self.records]
            ),
        }


def pair_manifest_content_hash(
    manifest: Mapping[str, Any],
    records: Sequence[PairRecord],
) -> str:
    """Return a stable content hash for a frozen pair manifest.

    The hash intentionally excludes timestamp-style provenance such as
    ``generated_at`` and includes the ordered pair rows.  It is therefore stable
    across rematerialization when the dataset, split, policy, recipe, seed, and
    ordered scientific pair content are unchanged.
    """
    stable_manifest = dict(manifest)
    stable_manifest.pop("generated_at", None)
    stable_manifest.pop("content_identity", None)
    record_payloads = []
    for record in records:
        row = record.to_dict()
        if row.get("distance_bin_label") is None:
            row.pop("distance_bin_label", None)
        record_payloads.append(row)
    return _stable_sha256(
        {
            "manifest": stable_manifest,
            "records": record_payloads,
        }
    )


class PairSampler:
    """Sample ordered pairs from indexed catalog groups without O(N^2) tables."""

    catalog: SampleCatalog
    split_registry: SplitRegistry
    policy: PairPolicy

    def __init__(
        self,
        catalog: SampleCatalog,
        split_registry: SplitRegistry,
        policy: PairPolicy | None = None,
    ) -> None:
        split_registry.validate_catalog(catalog)
        self.catalog = catalog
        self.split_registry = split_registry
        self.policy = PairPolicy() if policy is None else policy
        self._eligible_cache: dict[tuple[str, str, str, str], np.ndarray] = {}
        self._bucket_cache: dict[tuple[str, str, str, str], list[np.ndarray]] = {}
        self._validate_policy_against_catalog(self.policy)

    def _validate_policy_against_catalog(self, policy: PairPolicy) -> None:
        catalog_families = {str(v) for v in self.catalog.dataset_families}
        requested = set(str(v) for v in policy.dataset_families)
        weighted = set(str(v) for v in policy.dataset_family_weights)
        sampled = set(str(v) for v in policy.dataset_family_sampling)
        unknown = sorted((requested | weighted | sampled) - catalog_families)
        if unknown:
            raise ValueError(
                f"Pair policy {policy.policy_id!r} references unknown dataset family "
                f"name(s) {unknown}; available={sorted(catalog_families)}."
            )
        if policy.same_pair_id and not any(str(v) for v in self.catalog.pair_ids):
            raise ValueError(
                f"Pair policy {policy.policy_id!r} sets same_pair_id=True, but the "
                "catalog has no non-empty pair_id values. V4 policies should normally "
                "use same_pair_id=False."
            )

    def eligible_indices(
        self,
        science_split: str,
        nuisance_split: str,
        *,
        policy: PairPolicy | None = None,
        dataset_family: str | None = None,
    ) -> np.ndarray:
        """Return samples whose science and nuisance groups match split choices."""
        policy = self.policy if policy is None else policy
        family_key = "" if dataset_family is None else str(dataset_family)
        key = (
            str(science_split),
            str(nuisance_split),
            _stable_sha256(policy.to_dict()),
            family_key,
        )
        cached = self._eligible_cache.get(key)
        if cached is not None:
            return cached
        science_groups = self.split_registry.science_groups(str(science_split))
        mask = np.asarray([str(v) in science_groups for v in self.catalog.science_group_ids])
        if str(nuisance_split) not in {"*", "all", "any"}:
            nuisance_groups = self.split_registry.nuisance_groups(str(nuisance_split))
            mask &= np.asarray(
                [str(v) in nuisance_groups for v in self.catalog.nuisance_group_ids]
            )
        if dataset_family is not None:
            mask &= np.asarray(
                [str(v) == str(dataset_family) for v in self.catalog.dataset_families]
            )
        elif policy.dataset_families:
            allowed = set(policy.dataset_families)
            mask &= np.asarray([str(v) in allowed for v in self.catalog.dataset_families])
        if policy.joint_train_prefix_size is not None:
            prefix = int(policy.joint_train_prefix_size)
            is_joint_train = (
                (self.catalog.dataset_families == "joint_full_v4")
                & (self.catalog.sample_roles == "train")
            )
            prefix_ok = self.catalog.joint_train_sequence_indices >= 0
            prefix_ok &= self.catalog.joint_train_sequence_indices < prefix
            mask &= (~is_joint_train) | prefix_ok
        indices = np.flatnonzero(mask).astype(np.int64)
        self._eligible_cache[key] = indices
        return indices

    def _buckets(
        self,
        *,
        family: str,
        science_split: str,
        nuisance_split: str,
        policy: PairPolicy,
        dataset_family: str | None,
    ) -> list[np.ndarray]:
        family = _canonical_family(family)
        cache_key = (
            family,
            str(science_split),
            str(nuisance_split),
            _stable_sha256({"policy": policy.to_dict(), "dataset_family": dataset_family}),
        )
        cached = self._bucket_cache.get(cache_key)
        if cached is not None:
            return cached
        indices = self.eligible_indices(
            science_split,
            nuisance_split,
            policy=policy,
            dataset_family=dataset_family,
        )
        buckets: dict[tuple[str, ...], list[int]] = {}
        for idx in indices:
            i = int(idx)
            pair_part = (str(self.catalog.pair_ids[i]),) if policy.same_pair_id else ()
            if policy.same_pair_id and not pair_part[0]:
                continue
            if family == "same_nuisance_different_science":
                key = (str(self.catalog.nuisance_group_ids[i]), *pair_part)
            elif family in {"same_science_different_nuisance", "different_nuisance_same_science"}:
                key = (str(self.catalog.science_group_ids[i]), *pair_part)
            elif family == "different_science_different_nuisance":
                key = (*pair_part,)
            elif family == "identity":
                key = (str(self.catalog.science_group_ids[i]), str(self.catalog.nuisance_group_ids[i]), *pair_part)
            else:
                raise ValueError(f"Unsupported pair family {family!r}.")
            buckets.setdefault(key, []).append(i)
        valid: list[np.ndarray] = []
        for values in buckets.values():
            arr = np.asarray(values, dtype=np.int64)
            if arr.size < 2 and family != "identity":
                continue
            if family == "same_nuisance_different_science":
                if len(set(str(self.catalog.science_group_ids[i]) for i in arr)) < 2:
                    continue
            elif family in {"same_science_different_nuisance", "different_nuisance_same_science"}:
                if len(set(str(self.catalog.nuisance_group_ids[i]) for i in arr)) < 2:
                    continue
            elif family == "different_science_different_nuisance":
                science = len(set(str(self.catalog.science_group_ids[i]) for i in arr))
                nuisance = len(set(str(self.catalog.nuisance_group_ids[i]) for i in arr))
                if science < 2 or nuisance < 2:
                    continue
            elif family == "identity":
                if arr.size < 1:
                    continue
            valid.append(arr)
        if not valid:
            raise ValueError(
                f"No candidate buckets for {family!r} in science_split={science_split!r}, "
                f"nuisance_split={nuisance_split!r}, dataset_family={dataset_family!r}."
            )
        self._bucket_cache[cache_key] = valid
        return valid

    def _choose_family(self, rng: np.random.Generator, policy: PairPolicy) -> str:
        return _canonical_family(_choose_weighted_key(rng, policy.family_weights))

    def _choose_dataset_family(
        self,
        rng: np.random.Generator,
        policy: PairPolicy,
    ) -> str | None:
        family_weights = policy.dataset_family_sampling_weights()
        if family_weights:
            return _choose_weighted_key(rng, family_weights)
        if policy.dataset_family_weights:
            return _choose_weighted_key(rng, policy.dataset_family_weights)
        return None

    def _pair_satisfies_family(self, a: int, b: int, family: str, policy: PairPolicy) -> bool:
        family = _canonical_family(family)
        same_science = str(self.catalog.science_group_ids[a]) == str(
            self.catalog.science_group_ids[b]
        )
        same_nuisance = str(self.catalog.nuisance_group_ids[a]) == str(
            self.catalog.nuisance_group_ids[b]
        )
        if a == b and family != "identity" and not policy.allow_identity_pairs:
            return False
        if policy.same_pair_id and str(self.catalog.pair_ids[a]) != str(
            self.catalog.pair_ids[b]
        ):
            return False
        if policy.require_same_dataset_family and str(self.catalog.dataset_families[a]) != str(
            self.catalog.dataset_families[b]
        ):
            return False
        if family == "same_nuisance_different_science":
            return same_nuisance and not same_science
        if family in {"same_science_different_nuisance", "different_nuisance_same_science"}:
            return same_science and not same_nuisance
        if family == "different_science_different_nuisance":
            return not same_science and not same_nuisance
        if family == "identity":
            return same_science and same_nuisance and a == b
        raise ValueError(f"Unsupported pair family {family!r}.")

    def _selected_distance_bin(self, rng: np.random.Generator, policy: PairPolicy) -> str | None:
        if policy.distance_bin_weights:
            return _choose_weighted_key(rng, policy.distance_bin_weights)
        if policy.fisher_distance_bins:
            return str(policy.fisher_distance_bins[int(rng.integers(0, len(policy.fisher_distance_bins)))])
        return None

    def _pair_distance_ok(
        self,
        a: int,
        b: int,
        *,
        policy: PairPolicy,
        required_bin: str | None,
    ) -> bool:
        delta = self.catalog.fisher_scaled_deltas[b] - self.catalog.fisher_scaled_deltas[a]
        distance = float(np.linalg.norm(delta))
        if distance < float(policy.min_fisher_distance):
            return False
        if policy.max_fisher_distance is not None and distance > float(
            policy.max_fisher_distance
        ):
            return False
        if required_bin is not None:
            observed = fisher_distance_bin_label(distance, policy.fisher_distance_bin_edges)
            if observed != str(required_bin):
                return False
        changed = int(np.count_nonzero(np.abs(delta) > 1.0e-12))
        if (
            policy.max_changed_science_dimensions is not None
            and changed > int(policy.max_changed_science_dimensions)
        ):
            return False
        return True

    def sample_pair(
        self,
        rng: np.random.Generator,
        *,
        science_split: str = "train",
        nuisance_split: str = "train",
        split: str = "train",
        eval_slice: str | None = None,
        epoch: int | None = None,
    ) -> PairRecord:
        """Sample one ordered pair respecting split, family, and distance policy."""
        policy = self.policy.resolved_for_epoch(epoch)
        self._validate_policy_against_catalog(policy)
        diagnostics: dict[str, Any] = {
            "policy_id": policy.policy_id,
            "science_split": science_split,
            "nuisance_split": nuisance_split,
            "attempted": [],
        }
        for _ in range(int(policy.max_sampling_attempts)):
            family = self._choose_family(rng, policy)
            dataset_family = self._choose_dataset_family(rng, policy)
            family_policy = policy.resolved_for_dataset_family(dataset_family)
            required_bin = self._selected_distance_bin(rng, family_policy)
            buckets = self._buckets(
                family=family,
                science_split=science_split,
                nuisance_split=nuisance_split,
                policy=family_policy,
                dataset_family=dataset_family,
            )
            diagnostics["attempted"].append(
                {
                    "pair_family": family,
                    "dataset_family": dataset_family,
                    "distance_bin": required_bin,
                    "candidate_bucket_count": len(buckets),
                    "candidate_sample_count": int(sum(len(bucket) for bucket in buckets)),
                }
            )
            bucket = buckets[int(rng.integers(0, len(buckets)))]
            a = int(bucket[int(rng.integers(0, len(bucket)))])
            b = a if family == "identity" else int(bucket[int(rng.integers(0, len(bucket)))])
            if not self._pair_satisfies_family(a, b, family, policy):
                continue
            if not self._pair_distance_ok(
                a,
                b,
                policy=family_policy,
                required_bin=required_bin,
            ):
                continue
            return self.make_pair_record(
                a,
                b,
                family=family,
                split=split,
                eval_slice=eval_slice,
                policy=policy,
            )
        raise RuntimeError(
            "Could not sample a valid pair after "
            f"{policy.max_sampling_attempts} attempts for policy {policy.policy_id!r}; "
            f"diagnostics={diagnostics}."
        )

    def make_pair_record(
        self,
        sample_a_index: int,
        sample_b_index: int,
        *,
        family: str,
        split: str,
        eval_slice: str | None,
        pair_record_id: str | None = None,
        policy: PairPolicy | None = None,
    ) -> PairRecord:
        """Build a fully populated ordered-pair record from two catalog indices."""
        policy = self.policy if policy is None else policy
        a = int(sample_a_index)
        b = int(sample_b_index)
        family = _canonical_family(family)
        if family in {"different_nuisance_same_science", "same_science_different_nuisance", "identity"}:
            delta_z = np.zeros_like(self.catalog.fisher_scaled_deltas[a])
            delta_theta = np.zeros_like(self.catalog.physical_deltas[a])
        else:
            delta_z = self.catalog.fisher_scaled_deltas[b] - self.catalog.fisher_scaled_deltas[a]
            delta_theta = self.catalog.physical_deltas[b] - self.catalog.physical_deltas[a]
        nuisance_delta = self.catalog.nuisance_vectors[b] - self.catalog.nuisance_vectors[a]
        distance = float(np.linalg.norm(delta_z))
        if pair_record_id is None:
            pair_record_id = _stable_id(
                [
                    self.catalog.prepared_dataset_hash,
                    self.split_registry.artifact_id,
                    policy.policy_id,
                    split,
                    eval_slice,
                    self.catalog.sample_ids[a],
                    self.catalog.sample_ids[b],
                ]
            )
        return PairRecord(
            pair_record_id=pair_record_id,
            sample_a_id=str(self.catalog.sample_ids[a]),
            sample_b_id=str(self.catalog.sample_ids[b]),
            sample_a_index=int(self.catalog.array_indices[a]),
            sample_b_index=int(self.catalog.array_indices[b]),
            target_delta_z=tuple(float(v) for v in delta_z),
            target_delta_theta=tuple(float(v) for v in delta_theta),
            nuisance_delta=tuple(float(v) for v in nuisance_delta),
            nuisance_a_id=str(self.catalog.nuisance_group_ids[a]),
            nuisance_b_id=str(self.catalog.nuisance_group_ids[b]),
            science_a_id=str(self.catalog.science_group_ids[a]),
            science_b_id=str(self.catalog.science_group_ids[b]),
            pair_family=family,
            split=str(split),
            eval_slice=eval_slice,
            fisher_distance_l2=distance,
            changed_science_dimensions=int(np.count_nonzero(np.abs(delta_z) > 1.0e-12)),
            dataset_family_a=str(self.catalog.dataset_families[a]),
            dataset_family_b=str(self.catalog.dataset_families[b]),
            pair_id_a=str(self.catalog.pair_ids[a]),
            pair_id_b=str(self.catalog.pair_ids[b]),
            prepared_dataset_hash=self.catalog.prepared_dataset_hash,
            split_registry_id=self.split_registry.artifact_id,
            pair_policy_id=policy.policy_id,
            distance_bin_label=fisher_distance_bin_label(
                distance,
                policy.fisher_distance_bin_edges,
            ),
        )


def generate_frozen_pair_manifest(
    catalog: SampleCatalog,
    split_registry: SplitRegistry,
    *,
    policy: PairPolicy | None = None,
    artifact_id: str = DEFAULT_PAIR_ARTIFACT_ID,
    split: str = "validation",
    seed: int = 0,
    pairs_per_slice: int = 256,
    eval_slices: Mapping[str, Mapping[str, str]] | None = None,
) -> PairManifest:
    """Generate deterministic frozen validation/test ordered-pair records."""
    if int(pairs_per_slice) < 1:
        raise ValueError("pairs_per_slice must be >= 1.")
    policy = PairPolicy() if policy is None else policy
    if eval_slices is None:
        eval_slices = {
            "heldout_science_seen_nuisance": {
                "science_split": split,
                "nuisance_split": "train",
            },
            "heldout_science_heldout_nuisance": {
                "science_split": split,
                "nuisance_split": split,
            },
        }
    sampler = PairSampler(catalog, split_registry, policy)
    records: list[PairRecord] = []
    rng = np.random.default_rng(int(seed))
    seen_ids: set[str] = set()
    sample_lookup = catalog.sample_id_to_index
    for slice_name, selection in eval_slices.items():
        slice_count = 0
        attempts = 0
        max_attempts = int(pairs_per_slice) * int(policy.max_sampling_attempts)
        while slice_count < int(pairs_per_slice) and attempts < max_attempts:
            attempts += 1
            record = sampler.sample_pair(
                rng,
                science_split=str(selection["science_split"]),
                nuisance_split=str(selection["nuisance_split"]),
                split=split,
                eval_slice=str(slice_name),
            )
            record_id = _stable_id(
                [
                    artifact_id,
                    split,
                    slice_name,
                    seed,
                    slice_count,
                    record.sample_a_id,
                    record.sample_b_id,
                ],
                prefix="eval_pair",
            )
            record = sampler.make_pair_record(
                sample_lookup[record.sample_a_id],
                sample_lookup[record.sample_b_id],
                family=record.pair_family,
                split=split,
                eval_slice=str(slice_name),
                pair_record_id=record_id,
                policy=policy,
            )
            if record.pair_record_id in seen_ids:
                continue
            seen_ids.add(record.pair_record_id)
            records.append(record)
            slice_count += 1
            if policy.include_reverse:
                reverse_id = _stable_id(
                    [record_id, "reverse", record.sample_b_id, record.sample_a_id],
                    prefix="eval_pair",
                )
                reverse = make_reverse_pair_record(
                    record,
                    pair_record_id=reverse_id,
                )
                if reverse.pair_record_id not in seen_ids:
                    seen_ids.add(reverse.pair_record_id)
                    records.append(reverse)
        if slice_count < int(pairs_per_slice):
            raise RuntimeError(
                f"Could only generate {slice_count} unique pairs for eval slice {slice_name!r}; "
                f"requested {pairs_per_slice}."
            )
    manifest = {
        "schema_version": PAIR_MANIFEST_SCHEMA_VERSION,
        "artifact_id": str(artifact_id),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "prepared_dataset": {
            "artifact_id": catalog.artifact_id,
            "prepared_dataset_hash": catalog.prepared_dataset_hash,
            "sample_count": catalog.sample_count,
            "science_dim": catalog.science_dim,
        },
        "split_registry": {
            "artifact_id": split_registry.artifact_id,
            "prepared_dataset_hash": split_registry.prepared_dataset.get(
                "prepared_dataset_hash"
            ),
            "content_sha256": split_registry_content_sha256(split_registry),
        },
        "split": str(split),
        "seed": int(seed),
        "pair_count": len(records),
        "pairs_per_slice_requested": int(pairs_per_slice),
        "eval_slices": {key: dict(value) for key, value in eval_slices.items()},
        "pair_policy": policy.to_dict(),
        "pair_family_counts": _counts([record.pair_family for record in records]),
        "eval_slice_counts": _counts([record.eval_slice or "none" for record in records]),
        "distance_summary": _distance_summary([record.fisher_distance_l2 for record in records]),
        "target_convention": "target_delta_z = z_B - z_A",
    }
    manifest["content_identity"] = {
        "algorithm": "sha256/json-canonical/pair-manifest-v1",
        "sha256": pair_manifest_content_hash(manifest, records),
        "excludes": ["generated_at"],
    }
    return PairManifest(artifact_id=str(artifact_id), manifest=manifest, records=tuple(records))


def write_pair_manifest(path: Path, pair_manifest: PairManifest, *, overwrite: bool = False) -> None:
    """Write a frozen pair manifest directory with metadata and JSONL rows."""
    root = Path(path)
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(f"{root} exists and is non-empty; pass overwrite=True.")
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "manifest.json", pair_manifest.manifest)
    write_jsonl(root / "pairs.jsonl", (record.to_dict() for record in pair_manifest.records))


def load_pair_manifest(
    path: Path,
    *,
    catalog: SampleCatalog | None = None,
    split_registry: SplitRegistry | None = None,
) -> PairManifest:
    """Load a frozen pair manifest and optionally validate identity links."""
    root = Path(path)
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema_version") != PAIR_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported pair manifest schema {manifest.get('schema_version')!r}.")
    records = tuple(PairRecord.from_dict(row) for row in read_jsonl(root / "pairs.jsonl"))
    if len(records) != int(manifest.get("pair_count", len(records))):
        raise ValueError("Pair manifest pair_count does not match pairs.jsonl row count.")
    content_identity = manifest.get("content_identity", {})
    if isinstance(content_identity, Mapping) and content_identity.get("sha256"):
        actual = pair_manifest_content_hash(manifest, records)
        if str(content_identity["sha256"]) != actual:
            raise ValueError(
                "Pair manifest content hash does not match pairs.jsonl and manifest metadata "
                f"({content_identity['sha256']} != {actual})."
            )
    if catalog is not None:
        expected = manifest.get("prepared_dataset", {}).get("prepared_dataset_hash")
        if expected != catalog.prepared_dataset_hash:
            raise ValueError(
                "Pair manifest was generated for a different prepared dataset "
                f"({expected} != {catalog.prepared_dataset_hash})."
            )
    if split_registry is not None:
        expected = manifest.get("split_registry", {}).get("artifact_id")
        if expected != split_registry.artifact_id:
            raise ValueError(
                f"Pair manifest split registry {expected!r} does not match {split_registry.artifact_id!r}."
            )
        expected_content = manifest.get("split_registry", {}).get("content_sha256")
        actual_content = split_registry_content_sha256(split_registry)
        if expected_content and str(expected_content) != actual_content:
            raise ValueError(
                "Pair manifest split registry content hash does not match current split "
                f"({expected_content} != {actual_content})."
            )
    return PairManifest(
        artifact_id=str(manifest["artifact_id"]),
        manifest=manifest,
        records=records,
    )
