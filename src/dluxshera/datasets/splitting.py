from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .schema import json_ready

__all__ = ["GroupedSplitResult", "assign_grouped_split"]

Record = Mapping[str, Any]
GroupKeySpec = str | Sequence[str] | Callable[[Record], Any]


@dataclass(frozen=True)
class GroupedSplitResult:
    """Hold deterministic group-level and record-level split assignments."""

    record_assignments: tuple[str, ...]
    group_assignments: Mapping[str, str]
    policy: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "record_assignments": list(self.record_assignments),
            "group_assignments": dict(self.group_assignments),
            "policy": json_ready(dict(self.policy)),
        }


def _normalize_fractions(fractions: Mapping[str, float]) -> tuple[list[str], list[float]]:
    if not fractions:
        raise ValueError("fractions must contain at least one partition.")
    names: list[str] = []
    values: list[float] = []
    for name, value in fractions.items():
        partition = str(name).strip()
        if not partition:
            raise ValueError("partition names must be non-empty.")
        if partition in names:
            raise ValueError(f"Duplicate partition name {partition!r}.")
        fraction = float(value)
        if fraction < 0 or not math.isfinite(fraction):
            raise ValueError("split fractions must be finite and >= 0.")
        names.append(partition)
        values.append(fraction)
    total = sum(values)
    if total <= 0:
        raise ValueError("At least one split fraction must be positive.")
    return names, [value / total for value in values]


def _stable_group_id(
    record: Record,
    group_keys: GroupKeySpec,
    *,
    allow_missing: bool,
) -> str:
    if callable(group_keys):
        raw = group_keys(record)
    elif isinstance(group_keys, str):
        if group_keys not in record and not allow_missing:
            raise KeyError(f"Record is missing requested group key {group_keys!r}.")
        raw = record.get(group_keys)
    else:
        raw = {}
        for key in group_keys:
            field = str(key)
            if field not in record and not allow_missing:
                raise KeyError(f"Record is missing requested group key {field!r}.")
            raw[field] = record.get(field)
    return json.dumps(json_ready(raw), sort_keys=True, separators=(",", ":"))


def _ordered_group_ids(
    records: Sequence[Record],
    group_keys: GroupKeySpec,
    *,
    allow_missing: bool,
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for record in records:
        group_id = _stable_group_id(record, group_keys, allow_missing=allow_missing)
        if group_id in seen:
            continue
        seen.add(group_id)
        out.append(group_id)
    return out


def _seeded_order(group_ids: Sequence[str], seed: int) -> list[str]:
    decorated: list[tuple[str, str]] = []
    for group_id in group_ids:
        payload = json.dumps([int(seed), group_id], separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        decorated.append((digest, group_id))
    decorated.sort(key=lambda item: item[0])
    return [group_id for _, group_id in decorated]


def _target_counts(n_groups: int, fractions: Sequence[float]) -> list[int]:
    raw = [n_groups * fraction for fraction in fractions]
    counts = [int(math.floor(value)) for value in raw]
    remaining = n_groups - sum(counts)
    order = sorted(
        range(len(raw)),
        key=lambda idx: (raw[idx] - counts[idx], -idx),
        reverse=True,
    )
    for idx in order[:remaining]:
        counts[idx] += 1
    return counts


def assign_grouped_split(
    records: Sequence[Record],
    *,
    group_keys: GroupKeySpec,
    fractions: Mapping[str, float],
    seed: int = 0,
    allow_missing_group_keys: bool = False,
    policy_name: str | None = None,
) -> GroupedSplitResult:
    """Assign records to deterministic partitions while keeping groups intact.

    The assignment happens at group granularity.  Rounding uses largest
    remainders on the requested group counts, then a stable seed-derived hash
    order assigns whole groups to partitions.

    Parameters
    ----------
    records:
        Ordered records to assign.
    group_keys:
        One field name, a sequence of field names, or a callable returning a
        group identity.  Missing named fields raise by default.
    fractions:
        Mapping from partition name to group-count fraction.  Fractions apply
        to the number of groups, not the number of records.
    seed:
        Stable seed used to hash-order groups before assignment.
    allow_missing_group_keys:
        Opt-in compatibility mode that groups missing named fields as ``None``.
    policy_name:
        Required stable name when ``group_keys`` is callable; copied into
        provenance instead of attempting to serialize Python code.
    """
    if callable(group_keys) and not policy_name:
        raise ValueError("policy_name is required when group_keys is callable.")
    partition_names, normalized_fractions = _normalize_fractions(fractions)
    group_ids = _ordered_group_ids(
        records,
        group_keys,
        allow_missing=bool(allow_missing_group_keys),
    )
    ordered = _seeded_order(group_ids, int(seed))
    counts = _target_counts(len(group_ids), normalized_fractions)

    group_assignments: dict[str, str] = {}
    cursor = 0
    for partition, count in zip(partition_names, counts):
        for group_id in ordered[cursor : cursor + count]:
            group_assignments[group_id] = partition
        cursor += count

    record_assignments = tuple(
        group_assignments[
            _stable_group_id(
                record,
                group_keys,
                allow_missing=bool(allow_missing_group_keys),
            )
        ]
        for record in records
    )
    return GroupedSplitResult(
        record_assignments=record_assignments,
        group_assignments=group_assignments,
        policy={
            "type": "grouped_split/1",
            "policy_name": policy_name,
            "group_keys": (
                group_keys
                if isinstance(group_keys, str)
                else list(group_keys)
                if not callable(group_keys)
                else f"callable:{policy_name}"
            ),
            "fractions": dict(fractions),
            "normalized_fractions": dict(zip(partition_names, normalized_fractions)),
            "seed": int(seed),
            "group_count": len(group_ids),
            "rounding": "largest_remainder",
            "fraction_basis": "groups",
            "allow_missing_group_keys": bool(allow_missing_group_keys),
        },
    )
