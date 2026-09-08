from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from dluxshera.datasets.schema import VectorSpaceSpec, json_ready, read_json, read_jsonl

__all__ = [
    "DATASET_VERSION",
    "DEFAULT_SHARD_SIZE",
    "PRODUCTION_IDENTITIES",
    "FrozenMasterV4",
    "FrozenV4Identities",
    "ResolvedRenderState",
    "SlurmArrayPlan",
    "array_task_range",
    "build_slurm_array_plan",
    "content_hash",
    "file_sha256",
    "locate_science_global_index",
    "render_index_location",
    "render_index_to_science_nuisance",
    "render_output_paths",
    "render_state_id",
    "science_nuisance_to_render_index",
    "shard_name_for_render_index",
]

DATASET_VERSION = "shera_ml_master_v4"
DEFAULT_SHARD_SIZE = 1000


@dataclass(frozen=True)
class FrozenV4Identities:
    """Hold the immutable frozen V4 scientific identity gates."""

    master_scientific_content_hash: str | None = None
    science_vector_space_id: str | None = None
    nuisance_vector_space_id: str | None = None
    nuisance_bank_hash: str | None = None
    render_system_contract_hash: str | None = None
    render_contract_hash: str | None = None


PRODUCTION_IDENTITIES = FrozenV4Identities(
    master_scientific_content_hash=(
        "dcc46d19cd87a65c2321b2586543c30478d1917b597dd8097397adcebea43fe4"
    ),
    science_vector_space_id=(
        "916a005234f944569f9e635873da6b79a6df505e26fcfd002ffb389cdec19f1b"
    ),
    nuisance_vector_space_id=(
        "abe2f0435286941c60ccadb4a0601fb2ff4dca5b470c5aeb31fec6f77357a85c"
    ),
    nuisance_bank_hash=(
        "1eb0ffe3058f4416ac0cc732414956cc9ccb55ea126c5405544003ef5d288a56"
    ),
    render_system_contract_hash=(
        "26fdb799813affefad51585c074f24f32841501036007992b9dfdebdfeb1deab"
    ),
    render_contract_hash=(
        "566d433b3510fccc6d91b983b9d03a20469e2400039e24987c0e1714b7b7ba8c"
    ),
)


@dataclass(frozen=True)
class ResolvedRenderState:
    """Describe one fully resolved frozen V4 render state."""

    render_index: int
    science_global_index: int
    nuisance_bank_index: int
    dataset_family: str
    split_role: str
    science_plan_row_index: int
    science_state_id: str
    nuisance_state_id: str
    render_state_id: str
    science_labels: tuple[str, ...]
    physical_science_vector: tuple[float, ...]
    fisher_science_vector: tuple[float, ...]
    nuisance_labels: tuple[str, ...]
    physical_nuisance_vector: tuple[float, ...]
    science_row: Mapping[str, Any]
    nuisance_row: Mapping[str, Any]


@dataclass(frozen=True)
class SlurmArrayPlan:
    """Summarize a stop-exclusive render-index Slurm array partition."""

    render_count: int
    renders_per_task: int
    task_count: int
    final_task_size: int
    array_expression: str
    concurrency: int | None = None
    nuisance_count: int | None = None
    alignment_warning: str | None = None


def canonical_json(value: Any) -> str:
    return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    """Return the repository-standard SHA256 hash for canonical JSON content."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Return a SHA256 hash for a file without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_state_id(science_id: str, nuisance_id: str, render_contract_id: str) -> str:
    """Return the frozen V4 render-state content identity."""

    return "render_" + content_hash(
        {
            "science_state_id": science_id,
            "nuisance_state_id": nuisance_id,
            "render_model_system_contract_identity": render_contract_id,
        }
    )[:32]


def science_nuisance_to_render_index(
    science_global_index: int,
    nuisance_bank_index: int,
    nuisance_count: int,
) -> int:
    if science_global_index < 0:
        raise ValueError("science_global_index must be non-negative.")
    if nuisance_count <= 0:
        raise ValueError("nuisance_count must be positive.")
    if nuisance_bank_index < 0 or nuisance_bank_index >= nuisance_count:
        raise ValueError("nuisance_bank_index must be in [0, nuisance_count).")
    return int(science_global_index) * int(nuisance_count) + int(nuisance_bank_index)


def render_index_to_science_nuisance(
    render_index: int,
    nuisance_count: int,
) -> tuple[int, int]:
    if render_index < 0:
        raise ValueError("render_index must be non-negative.")
    if nuisance_count <= 0:
        raise ValueError("nuisance_count must be positive.")
    return divmod(int(render_index), int(nuisance_count))


def locate_science_global_index(
    render_contract: Mapping[str, Any],
    science_global_index: int,
) -> dict[str, Any]:
    idx = int(science_global_index)
    for entry in render_contract["families"]:
        start = int(entry["science_start_index"])
        count = int(entry["science_count"])
        if start <= idx < start + count:
            return {
                "family": str(entry["family"]),
                "split_role": str(entry["split_role"]),
                "science_global_index": idx,
                "science_plan_row_index": idx - start,
            }
    raise ValueError(f"science_global_index {idx} is outside the render contract science range.")


def render_index_location(
    render_contract: Mapping[str, Any],
    render_index: int,
) -> dict[str, Any]:
    render_count = int(render_contract["render_count"])
    if int(render_index) < 0 or int(render_index) >= render_count:
        raise ValueError(
            f"render_index {render_index} is outside [0, {render_count})."
        )
    science_global_index, nuisance_bank_index = render_index_to_science_nuisance(
        int(render_index),
        int(render_contract["nuisance_count"]),
    )
    location = locate_science_global_index(render_contract, science_global_index)
    location["render_index"] = int(render_index)
    location["nuisance_bank_index"] = nuisance_bank_index
    return location


def shard_name_for_render_index(render_index: int, *, shard_size: int = DEFAULT_SHARD_SIZE) -> str:
    """Return the deterministic output shard name for a render index."""

    if shard_size <= 0:
        raise ValueError("shard_size must be positive.")
    if render_index < 0:
        raise ValueError("render_index must be non-negative.")
    return f"shard_{int(render_index) // int(shard_size):04d}"


def render_output_paths(
    *,
    output_root: Path,
    dataset_family: str,
    split_role: str,
    render_index: int,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> tuple[Path, Path]:
    """Return deterministic FITS and JSON paths for one raw V4 render."""

    stem = f"render_{int(render_index):07d}"
    root = (
        Path(output_root)
        / "images"
        / str(dataset_family)
        / str(split_role)
        / shard_name_for_render_index(render_index, shard_size=shard_size)
    )
    return root / f"{stem}.fits", root / f"{stem}.json"


def _expect_equal(name: str, actual: Any, expected: str | None) -> None:
    if expected is None:
        return
    if str(actual) != expected:
        raise ValueError(f"Frozen V4 contract mismatch for {name}: expected {expected}, got {actual}.")


def _validate_payload_hash(path: Path, payload: Mapping[str, Any], key: str) -> None:
    recorded = payload.get(key)
    if recorded is None:
        raise ValueError(f"{path} is missing required {key!r}.")
    recomputed = content_hash({k: v for k, v in payload.items() if k != key})
    if str(recorded) != recomputed:
        raise ValueError(
            f"{path} {key} mismatch: recorded {recorded}, recomputed {recomputed}."
        )


def _science_space(vector_spaces: Mapping[str, Any]) -> VectorSpaceSpec:
    return VectorSpaceSpec.from_dict(vector_spaces["spaces"]["physical_science_state"])


def _nuisance_space(vector_spaces: Mapping[str, Any]) -> VectorSpaceSpec:
    return VectorSpaceSpec.from_dict(vector_spaces["spaces"]["registration_nuisance"])


def _component_indices(space: VectorSpaceSpec) -> dict[str, int | None]:
    return {component.label: component.component_index for component in space.components}


def _read_jsonl_rows(path: Path, row_indices: Iterable[int]) -> dict[int, dict[str, Any]]:
    wanted = sorted({int(index) for index in row_indices})
    if not wanted:
        return {}
    if wanted[0] < 0:
        raise ValueError("JSONL row indices must be non-negative.")
    stop = wanted[-1]
    rows: dict[int, dict[str, Any]] = {}
    wanted_set = set(wanted)
    for idx, row in enumerate(read_jsonl(path)):
        if idx in wanted_set:
            rows[idx] = row
            if len(rows) == len(wanted):
                break
        if idx > stop:
            break
    missing = [idx for idx in wanted if idx not in rows]
    if missing:
        raise ValueError(f"{path} is missing required state-plan row(s): {missing}.")
    return rows


class FrozenMasterV4:
    """Read and validate frozen SHERA ML Master V4 state plans."""

    def __init__(
        self,
        plan_root: Path,
        *,
        expected: FrozenV4Identities | None = PRODUCTION_IDENTITIES,
        validate_plan_hashes: bool = True,
    ) -> None:
        self.plan_root = Path(plan_root)
        self.manifest = read_json(self.plan_root / "freeze_manifest.json")
        self.vector_spaces = read_json(self.plan_root / "vector_spaces.json")
        self.nuisance_bank = read_json(self.plan_root / "nuisance_bank.json")
        self.render_system_contract = read_json(
            self.plan_root / "render_system_contract.json"
        )
        self.render_contract = read_json(self.plan_root / "render_contract.json")
        self._validate_contract(expected=expected, validate_plan_hashes=validate_plan_hashes)
        self.science_space = _science_space(self.vector_spaces)
        self.nuisance_space = _nuisance_space(self.vector_spaces)
        self.science_labels = self.science_space.labels
        self.nuisance_labels = self.nuisance_space.labels
        self.component_indices = _component_indices(self.science_space)
        self.nuisance_by_index = {
            int(row["bank_index"]): dict(row)
            for row in self.nuisance_bank["states"]
        }

    @property
    def render_count(self) -> int:
        return int(self.render_contract["render_count"])

    @property
    def nuisance_count(self) -> int:
        return int(self.render_contract["nuisance_count"])

    def _validate_contract(
        self,
        *,
        expected: FrozenV4Identities | None,
        validate_plan_hashes: bool,
    ) -> None:
        expected = FrozenV4Identities() if expected is None else expected
        _expect_equal(
            "master_scientific_content_hash",
            self.manifest.get("master_scientific_content_hash"),
            expected.master_scientific_content_hash,
        )
        _expect_equal(
            "science_vector_space_id",
            self.manifest.get("science_vector_space_id"),
            expected.science_vector_space_id,
        )
        _expect_equal(
            "nuisance_vector_space_id",
            self.manifest.get("nuisance_vector_space_id"),
            expected.nuisance_vector_space_id,
        )
        _expect_equal(
            "nuisance_bank_hash",
            self.manifest.get("selected_nuisance_bank_hash"),
            expected.nuisance_bank_hash,
        )
        _expect_equal(
            "render_system_contract_hash",
            self.manifest.get("render_system_contract_hash"),
            expected.render_system_contract_hash,
        )
        _expect_equal(
            "render_contract_hash",
            self.manifest.get("render_contract_hash"),
            expected.render_contract_hash,
        )
        if "scientific_identity" in self.manifest:
            master_hash = content_hash(self.manifest["scientific_identity"])
            if master_hash != self.manifest.get("master_scientific_content_hash"):
                raise ValueError(
                    "freeze_manifest.json master scientific hash does not match "
                    "its scientific_identity payload."
                )
        if "content_hash" in self.vector_spaces:
            _validate_payload_hash(
                self.plan_root / "vector_spaces.json",
                self.vector_spaces,
                "content_hash",
            )
        if self.nuisance_bank.get("content_hash") != self.manifest.get("selected_nuisance_bank_hash"):
            raise ValueError("nuisance_bank.json content_hash does not match freeze_manifest.json.")
        if self.render_system_contract.get("content_hash") != self.manifest.get("render_system_contract_hash"):
            raise ValueError("render_system_contract.json content_hash does not match freeze_manifest.json.")
        if self.render_contract.get("content_hash") != self.manifest.get("render_contract_hash"):
            raise ValueError("render_contract.json content_hash does not match freeze_manifest.json.")
        if self.render_contract.get("science_vector_space_id") != self.manifest.get("science_vector_space_id"):
            raise ValueError("render_contract.json science_vector_space_id does not match freeze_manifest.json.")
        if self.render_contract.get("nuisance_vector_space_id") != self.manifest.get("nuisance_vector_space_id"):
            raise ValueError("render_contract.json nuisance_vector_space_id does not match freeze_manifest.json.")
        if validate_plan_hashes:
            for entry in self.render_contract["families"]:
                plan_path = self.plan_root / str(entry["plan_path"])
                actual = file_sha256(plan_path)
                expected_hash = str(entry["plan_sha256"])
                if actual != expected_hash:
                    raise ValueError(
                        f"State-plan hash mismatch for {entry['plan_path']}: "
                        f"expected {expected_hash}, got {actual}."
                    )

    def locations_for_indices(self, render_indices: Iterable[int]) -> list[dict[str, Any]]:
        """Return render locations in requested order after range validation."""

        return [
            render_index_location(self.render_contract, int(render_index))
            for render_index in render_indices
        ]

    def resolve_indices(self, render_indices: Iterable[int]) -> list[ResolvedRenderState]:
        """Resolve render indices to science rows and nuisance-bank rows efficiently."""

        locations = self.locations_for_indices(render_indices)
        rows_by_plan: dict[tuple[str, str], set[int]] = {}
        for location in locations:
            key = (str(location["family"]), str(location["split_role"]))
            rows_by_plan.setdefault(key, set()).add(int(location["science_plan_row_index"]))

        plan_rows: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
        for (family, split), indices in rows_by_plan.items():
            path = self.plan_root / "state_plans" / family / f"{split}.jsonl"
            plan_rows[(family, split)] = _read_jsonl_rows(path, indices)

        resolved = []
        for location in locations:
            family = str(location["family"])
            split = str(location["split_role"])
            row_index = int(location["science_plan_row_index"])
            science = plan_rows[(family, split)][row_index]
            nuisance_index = int(location["nuisance_bank_index"])
            nuisance = self.nuisance_by_index[nuisance_index]
            expected_render_state_id = render_state_id(
                str(science["science_state_id"]),
                str(nuisance["nuisance_state_id"]),
                str(self.render_contract["render_system_contract_hash"]),
            )
            physical_science = tuple(
                float(value) for value in science["ordered_physical_science_vector"]
            )
            fisher_science = tuple(
                float(value) for value in science["ordered_fisher_scaled_delta"]
            )
            physical_nuisance = tuple(
                float(value) for value in nuisance["ordered_physical_nuisance_vector"]
            )
            self.science_space.validate_vector(physical_science, name="physical science vector")
            self.science_space.validate_vector(fisher_science, name="Fisher science vector")
            self.nuisance_space.validate_vector(physical_nuisance, name="physical nuisance vector")
            if str(science["science_vector_space_id"]) != str(self.manifest["science_vector_space_id"]):
                raise ValueError("Science row vector-space ID does not match frozen manifest.")
            if str(nuisance["nuisance_vector_space_id"]) != str(self.manifest["nuisance_vector_space_id"]):
                raise ValueError("Nuisance row vector-space ID does not match frozen manifest.")
            if str(science["dataset_family"]) != family or str(science["split_role"]) != split:
                raise ValueError("Science row family/split does not match render contract location.")
            resolved.append(
                ResolvedRenderState(
                    render_index=int(location["render_index"]),
                    science_global_index=int(location["science_global_index"]),
                    nuisance_bank_index=nuisance_index,
                    dataset_family=family,
                    split_role=split,
                    science_plan_row_index=row_index,
                    science_state_id=str(science["science_state_id"]),
                    nuisance_state_id=str(nuisance["nuisance_state_id"]),
                    render_state_id=expected_render_state_id,
                    science_labels=self.science_labels,
                    physical_science_vector=physical_science,
                    fisher_science_vector=fisher_science,
                    nuisance_labels=self.nuisance_labels,
                    physical_nuisance_vector=physical_nuisance,
                    science_row=science,
                    nuisance_row=nuisance,
                )
            )
        return resolved


def build_slurm_array_plan(
    *,
    render_count: int,
    renders_per_task: int,
    concurrency: int | None = None,
    nuisance_count: int | None = None,
) -> SlurmArrayPlan:
    """Return Slurm array geometry covering all render indices exactly once."""

    render_count = int(render_count)
    renders_per_task = int(renders_per_task)
    if render_count <= 0:
        raise ValueError("render_count must be positive.")
    if renders_per_task <= 0:
        raise ValueError("renders_per_task must be positive.")
    if concurrency is not None and int(concurrency) <= 0:
        raise ValueError("concurrency must be positive when provided.")
    task_count = int(math.ceil(render_count / renders_per_task))
    final_task_size = render_count - (task_count - 1) * renders_per_task
    suffix = "" if concurrency is None else f"%{int(concurrency)}"
    alignment_warning = None
    if nuisance_count is not None:
        nuisance_count = int(nuisance_count)
        if nuisance_count <= 0:
            raise ValueError("nuisance_count must be positive when provided.")
        if renders_per_task % nuisance_count != 0:
            alignment_warning = (
                "renders_per_task is not divisible by nuisance_count; "
                "array tasks may split science-state nuisance groups."
            )
    return SlurmArrayPlan(
        render_count=render_count,
        renders_per_task=renders_per_task,
        task_count=task_count,
        final_task_size=final_task_size,
        array_expression=f"0-{task_count - 1}{suffix}",
        concurrency=None if concurrency is None else int(concurrency),
        nuisance_count=nuisance_count,
        alignment_warning=alignment_warning,
    )


def array_task_range(
    *,
    task_id: int,
    render_count: int,
    renders_per_task: int,
) -> tuple[int, int]:
    """Return the stop-exclusive render range for one zero-based array task."""

    if task_id < 0:
        raise ValueError("task_id must be non-negative.")
    if render_count <= 0:
        raise ValueError("render_count must be positive.")
    if renders_per_task <= 0:
        raise ValueError("renders_per_task must be positive.")
    start = int(task_id) * int(renders_per_task)
    if start >= int(render_count):
        raise ValueError(f"task_id {task_id} starts beyond render_count {render_count}.")
    return start, min(start + int(renders_per_task), int(render_count))
