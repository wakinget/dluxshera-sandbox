from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dluxshera.datasets.schema import read_json, write_json

from .catalog import SampleCatalog

__all__ = [
    "SplitRegistry",
    "generate_split_registry",
    "load_split_registry",
    "write_split_registry",
]

SPLIT_SCHEMA_VERSION = "dluxshera_ml_split_registry/1"
DEFAULT_SPLIT_ARTIFACT_ID = "SPLIT-ML-v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    root = _repo_root()
    for key, cmd in {
        "commit": ["git", "-C", str(root), "rev-parse", "HEAD"],
        "branch": ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"],
        "dirty": ["git", "-C", str(root), "status", "--short"],
    }.items():
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            info[key] = None
        else:
            info[key] = bool(result.stdout.strip()) if key == "dirty" else result.stdout.strip()
    return info


def _seeded_order(values: list[str], seed: int, namespace: str) -> list[str]:
    decorated: list[tuple[str, str]] = []
    for value in values:
        payload = json.dumps([namespace, int(seed), value], separators=(",", ":"))
        decorated.append((hashlib.sha256(payload.encode("utf-8")).hexdigest(), value))
    decorated.sort(key=lambda item: item[0])
    return [value for _, value in decorated]


def _normalize_fractions(fractions: Mapping[str, float]) -> dict[str, float]:
    if not fractions:
        raise ValueError("fractions must not be empty.")
    cleaned: dict[str, float] = {}
    for key, value in fractions.items():
        name = str(key).strip()
        if not name:
            raise ValueError("split partition names must be non-empty.")
        fraction = float(value)
        if not np.isfinite(fraction) or fraction < 0.0:
            raise ValueError("split fractions must be finite and >= 0.")
        cleaned[name] = fraction
    total = sum(cleaned.values())
    if total <= 0.0:
        raise ValueError("At least one split fraction must be positive.")
    return {key: value / total for key, value in cleaned.items()}


def _target_counts(
    n_groups: int,
    fractions: Mapping[str, float],
    *,
    require_nonempty_requested: bool,
    group_kind: str,
) -> dict[str, int]:
    normalized = _normalize_fractions(fractions)
    positive = [name for name, value in normalized.items() if value > 0.0]
    if require_nonempty_requested and n_groups < len(positive):
        raise ValueError(
            f"Cannot assign non-empty {group_kind} partitions {positive} from only {n_groups} groups."
        )
    raw = {name: n_groups * fraction for name, fraction in normalized.items()}
    counts = {name: int(np.floor(value)) for name, value in raw.items()}
    remaining = int(n_groups - sum(counts.values()))
    order = sorted(
        normalized,
        key=lambda name: (raw[name] - counts[name], -list(normalized).index(name)),
        reverse=True,
    )
    for name in order[:remaining]:
        counts[name] += 1
    if require_nonempty_requested:
        for name in positive:
            if counts[name] == 0:
                donor = max(counts, key=lambda key: counts[key])
                if counts[donor] <= 1:
                    raise ValueError(
                        f"Cannot keep requested {group_kind} partition {name!r} non-empty."
                    )
                counts[donor] -= 1
                counts[name] += 1
    return counts


def _assign_groups(
    group_ids: list[str],
    *,
    fractions: Mapping[str, float],
    seed: int,
    namespace: str,
    require_nonempty_requested: bool,
    group_kind: str,
) -> dict[str, str]:
    ordered = _seeded_order(sorted(set(group_ids)), seed, namespace)
    counts = _target_counts(
        len(ordered),
        fractions,
        require_nonempty_requested=require_nonempty_requested,
        group_kind=group_kind,
    )
    assignments: dict[str, str] = {}
    cursor = 0
    for partition, count in counts.items():
        for group_id in ordered[cursor : cursor + int(count)]:
            assignments[str(group_id)] = str(partition)
        cursor += int(count)
    return assignments


def _counts(assignments: Mapping[str, str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for partition in assignments.values():
        out[str(partition)] = out.get(str(partition), 0) + 1
    return dict(sorted(out.items()))


@dataclass(frozen=True)
class SplitRegistry:
    """Represent reusable science-state and nuisance-state ML splits."""

    artifact_id: str
    schema_version: str
    prepared_dataset: Mapping[str, Any]
    seed: int
    science_group_policy: str
    nuisance_group_policy: str
    science_assignments: Mapping[str, str]
    nuisance_assignments: Mapping[str, str]
    policy: Mapping[str, Any]
    counts: Mapping[str, Any]
    generated_at: str
    git: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable split registry."""
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "prepared_dataset": dict(self.prepared_dataset),
            "seed": int(self.seed),
            "science_group_policy": self.science_group_policy,
            "nuisance_group_policy": self.nuisance_group_policy,
            "science_assignments": dict(sorted(self.science_assignments.items())),
            "nuisance_assignments": dict(sorted(self.nuisance_assignments.items())),
            "policy": dict(self.policy),
            "counts": dict(self.counts),
            "generated_at": self.generated_at,
            "git": dict(self.git),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SplitRegistry":
        """Build a split registry from serialized metadata."""
        if payload.get("schema_version") != SPLIT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported split registry schema {payload.get('schema_version')!r}."
            )
        return cls(
            artifact_id=str(payload["artifact_id"]),
            schema_version=str(payload["schema_version"]),
            prepared_dataset=dict(payload["prepared_dataset"]),
            seed=int(payload["seed"]),
            science_group_policy=str(payload["science_group_policy"]),
            nuisance_group_policy=str(payload["nuisance_group_policy"]),
            science_assignments=dict(payload["science_assignments"]),
            nuisance_assignments=dict(payload["nuisance_assignments"]),
            policy=dict(payload.get("policy", {})),
            counts=dict(payload.get("counts", {})),
            generated_at=str(payload.get("generated_at")),
            git=dict(payload.get("git", {})),
        )

    def science_split(self, group_id: str) -> str:
        """Return the split assignment for one science group."""
        return str(self.science_assignments[str(group_id)])

    def nuisance_split(self, group_id: str) -> str:
        """Return the split assignment for one nuisance group."""
        return str(self.nuisance_assignments[str(group_id)])

    def science_groups(self, split: str) -> set[str]:
        """Return science group ids assigned to ``split``."""
        return {group for group, part in self.science_assignments.items() if part == split}

    def nuisance_groups(self, split: str) -> set[str]:
        """Return nuisance group ids assigned to ``split``."""
        return {group for group, part in self.nuisance_assignments.items() if part == split}

    def validate_catalog(self, catalog: SampleCatalog) -> None:
        """Reject use with a different prepared dataset identity."""
        expected = self.prepared_dataset.get("prepared_dataset_hash")
        if expected != catalog.prepared_dataset_hash:
            raise ValueError(
                "Split registry was generated for a different prepared dataset "
                f"({expected} != {catalog.prepared_dataset_hash})."
            )


def generate_split_registry(
    catalog: SampleCatalog,
    *,
    artifact_id: str = DEFAULT_SPLIT_ARTIFACT_ID,
    seed: int = 0,
    science_fractions: Mapping[str, float] | None = None,
    nuisance_fractions: Mapping[str, float] | None = None,
    explicit_nuisance_assignments: Mapping[str, str] | None = None,
    require_nonempty_science_partitions: bool = True,
    require_nonempty_nuisance_partitions: bool = True,
) -> SplitRegistry:
    """Generate a deterministic reusable ML split registry.

    Science states and nuisance realizations are split independently.  The
    science grouping policy uses the prepared V3 physical-delta hash, keeping
    the same physical state together even when it appears in several pair-grid
    contexts.
    """
    science_fractions = dict(
        science_fractions or {"train": 0.8, "validation": 0.1, "test": 0.1}
    )
    nuisance_fractions = dict(
        nuisance_fractions or {"train": 0.8, "validation": 0.1, "test": 0.1}
    )
    science_ids = sorted(set(str(v) for v in catalog.science_group_ids))
    nuisance_ids = sorted(set(str(v) for v in catalog.nuisance_group_ids))
    science_assignments = _assign_groups(
        science_ids,
        fractions=science_fractions,
        seed=int(seed),
        namespace="science",
        require_nonempty_requested=bool(require_nonempty_science_partitions),
        group_kind="science",
    )
    if explicit_nuisance_assignments is not None:
        nuisance_assignments = {str(k): str(v) for k, v in explicit_nuisance_assignments.items()}
        missing = sorted(set(nuisance_ids) - set(nuisance_assignments))
        extra = sorted(set(nuisance_assignments) - set(nuisance_ids))
        if missing or extra:
            raise ValueError(
                "Explicit nuisance assignments must cover exactly the catalog nuisance ids; "
                f"missing={missing}, extra={extra}."
            )
        requested = set(str(v) for v in nuisance_assignments.values())
        if require_nonempty_nuisance_partitions:
            for partition in requested:
                if partition not in nuisance_assignments.values():
                    raise ValueError(f"Nuisance partition {partition!r} is empty.")
        nuisance_policy_type = "explicit"
    else:
        nuisance_assignments = _assign_groups(
            nuisance_ids,
            fractions=nuisance_fractions,
            seed=int(seed) + 104729,
            namespace="nuisance",
            require_nonempty_requested=bool(require_nonempty_nuisance_partitions),
            group_kind="nuisance",
        )
        nuisance_policy_type = "deterministic_fraction"

    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    return SplitRegistry(
        artifact_id=str(artifact_id),
        schema_version=SPLIT_SCHEMA_VERSION,
        prepared_dataset={
            "artifact_id": catalog.artifact_id,
            "prepared_dataset_hash": catalog.prepared_dataset_hash,
            "root": str(catalog.root),
            "sample_count": catalog.sample_count,
            "science_dim": catalog.science_dim,
            "sample_shape": list(catalog.sample_shape),
        },
        seed=int(seed),
        science_group_policy=catalog.science_group_policy,
        nuisance_group_policy=catalog.nuisance_group_policy,
        science_assignments=science_assignments,
        nuisance_assignments=nuisance_assignments,
        policy={
            "science": {
                "type": "deterministic_fraction",
                "fractions": science_fractions,
                "require_nonempty_requested": bool(require_nonempty_science_partitions),
            },
            "nuisance": {
                "type": nuisance_policy_type,
                "fractions": nuisance_fractions,
                "explicit_assignments_provided": explicit_nuisance_assignments is not None,
                "require_nonempty_requested": bool(require_nonempty_nuisance_partitions),
            },
        },
        counts={
            "science_groups": _counts(science_assignments),
            "nuisance_groups": _counts(nuisance_assignments),
            "sample_count": catalog.sample_count,
        },
        generated_at=generated_at,
        git=_git_info(),
    )


def load_split_registry(path: Path, *, catalog: SampleCatalog | None = None) -> SplitRegistry:
    """Load a split registry and optionally validate its prepared dataset."""
    registry = SplitRegistry.from_dict(read_json(Path(path)))
    if catalog is not None:
        registry.validate_catalog(catalog)
    return registry


def write_split_registry(path: Path, registry: SplitRegistry) -> None:
    """Write a split registry JSON artifact."""
    write_json(Path(path), registry.to_dict())
