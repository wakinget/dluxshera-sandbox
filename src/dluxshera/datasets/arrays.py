from __future__ import annotations

import json
import os
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

from .schema import json_ready, read_json, write_json

__all__ = ["ArrayShardReader", "ArrayShardStore", "ShardRecord"]

DEFAULT_TARGET_SHARD_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class ShardRecord:
    """Describe one fixed-shape array shard on disk."""

    shard_id: str
    path: str
    start_index: int
    sample_count: int
    sample_shape: tuple[int, ...]
    source_dtypes: tuple[str, ...]
    storage_dtype: str
    file_size_bytes: int

    @property
    def stop_index(self) -> int:
        """Return the exclusive global sample index for this shard."""
        return self.start_index + self.sample_count

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "shard_id": self.shard_id,
            "path": self.path,
            "start_index": self.start_index,
            "stop_index": self.stop_index,
            "sample_count": self.sample_count,
            "sample_shape": list(self.sample_shape),
            "source_dtypes": list(self.source_dtypes),
            "storage_dtype": self.storage_dtype,
            "file_size_bytes": self.file_size_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShardRecord":
        """Build a shard record from manifest metadata."""
        return cls(
            shard_id=str(payload["shard_id"]),
            path=str(payload["path"]),
            start_index=int(payload["start_index"]),
            sample_count=int(payload["sample_count"]),
            sample_shape=tuple(int(v) for v in payload["sample_shape"]),
            source_dtypes=tuple(str(v) for v in payload.get("source_dtypes", [])),
            storage_dtype=str(payload["storage_dtype"]),
            file_size_bytes=int(payload.get("file_size_bytes", 0)),
        )


def _dtype(value: Any) -> np.dtype:
    dtype = np.dtype(value)
    if dtype.hasobject:
        raise ValueError("Object dtype arrays are not supported by ArrayShardStore.")
    return dtype


def _samples_per_shard(
    *,
    sample_shape: tuple[int, ...],
    storage_dtype: np.dtype,
    target_shard_bytes: int,
    max_samples_per_shard: int | None,
) -> int:
    sample_bytes = int(np.prod(sample_shape, dtype=np.int64)) * int(storage_dtype.itemsize)
    if sample_bytes <= 0:
        raise ValueError("sample_shape must contain at least one element.")
    target_count = max(1, int(target_shard_bytes) // sample_bytes)
    if max_samples_per_shard is not None:
        if int(max_samples_per_shard) < 1:
            raise ValueError("max_samples_per_shard must be >= 1 when provided.")
        target_count = min(target_count, int(max_samples_per_shard))
    return max(1, target_count)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with tmp_path.open("wb") as handle:
        np.save(handle, array)
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


class ArrayShardStore:
    """Write fixed-shape arrays into deterministic, memory-map-friendly shards.

    The store accepts an iterable of per-sample arrays and writes bounded
    in-memory batches to ``.npy`` files.  It records source dtypes, explicit
    storage dtype, shard sizing policy, and sample-to-shard mappings in JSON
    artifacts.  It is generic array infrastructure and does not assume FITS,
    images, or ML training semantics.
    """

    output_dir: Path
    storage_dtype: np.dtype
    target_shard_bytes: int
    max_samples_per_shard: int | None
    manifest_name: str
    index_name: str

    def __init__(
        self,
        output_dir: Path,
        *,
        storage_dtype: Any = np.float32,
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
        max_samples_per_shard: int | None = None,
        manifest_name: str = "array_shards_manifest.json",
        index_name: str = "array_index.jsonl",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.storage_dtype = _dtype(storage_dtype)
        self.target_shard_bytes = int(target_shard_bytes)
        if self.target_shard_bytes < 1:
            raise ValueError("target_shard_bytes must be >= 1.")
        self.max_samples_per_shard = max_samples_per_shard
        self.manifest_name = str(manifest_name)
        self.index_name = str(index_name)

    def write(
        self,
        samples: Iterable[Any],
        *,
        sample_metadata: Iterable[Mapping[str, Any]] | None = None,
        extra_manifest: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write samples and return the shard manifest.

        ``sample_metadata`` is optional streaming metadata.  When supplied, one
        metadata row must be available for each sample; rows are copied into the
        generic index with shard id and offset fields appended.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        shards_dir = self.output_dir / "shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_dir / self.manifest_name
        index_path = self.output_dir / self.index_name
        index_tmp_path = index_path.with_name(index_path.name + ".tmp")
        if manifest_path.exists() or index_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing shard artifacts in {self.output_dir}."
            )
        if index_tmp_path.exists():
            index_tmp_path.unlink()
        for tmp_path in sorted(shards_dir.glob("*.tmp")):
            tmp_path.unlink()
        stale_shards = [path for path in shards_dir.iterdir() if path.exists()]
        if stale_shards:
            names = ", ".join(path.name for path in stale_shards[:5])
            suffix = "" if len(stale_shards) <= 5 else ", ..."
            raise FileExistsError(
                f"Refusing to write into non-empty shard directory {shards_dir}: {names}{suffix}"
            )

        metadata_iter = iter(sample_metadata) if sample_metadata is not None else None
        buffer: list[np.ndarray] = []
        buffer_source_dtypes: set[str] = set()
        buffer_source_dtype_values: list[str] = []
        shard_records: list[ShardRecord] = []
        all_source_dtypes: set[str] = set()
        sample_shape: tuple[int, ...] | None = None
        samples_per_shard: int | None = None
        sample_count = 0
        shard_index = 0

        created_shard_paths: list[Path] = []
        created_index_path = False
        try:
            with index_tmp_path.open("w", encoding="utf-8") as index_handle:
                for raw_sample in samples:
                    array = np.asarray(raw_sample)
                    if sample_shape is None:
                        sample_shape = tuple(int(v) for v in array.shape)
                        samples_per_shard = _samples_per_shard(
                            sample_shape=sample_shape,
                            storage_dtype=self.storage_dtype,
                            target_shard_bytes=self.target_shard_bytes,
                            max_samples_per_shard=self.max_samples_per_shard,
                        )
                    elif tuple(int(v) for v in array.shape) != sample_shape:
                        raise ValueError(
                            f"Sample {sample_count} has shape {array.shape}, expected {sample_shape}."
                        )

                    source_dtype = str(array.dtype)
                    all_source_dtypes.add(source_dtype)
                    buffer_source_dtypes.add(source_dtype)
                    buffer_source_dtype_values.append(source_dtype)
                    buffer.append(array.astype(self.storage_dtype, copy=False))

                    assert samples_per_shard is not None
                    if len(buffer) == samples_per_shard:
                        shard_record = self._flush_shard(
                            shards_dir=shards_dir,
                            shard_index=shard_index,
                            start_index=sample_count - len(buffer) + 1,
                            buffer=buffer,
                            source_dtypes=buffer_source_dtypes,
                        )
                        created_shard_paths.append(self.output_dir / shard_record.path)
                        shard_records.append(shard_record)
                        self._write_index_rows(
                            index_handle=index_handle,
                            shard_record=shard_record,
                            sample_source_dtypes=buffer_source_dtype_values,
                            sample_metadata_iter=metadata_iter,
                        )
                        shard_index += 1
                        buffer = []
                        buffer_source_dtypes = set()
                        buffer_source_dtype_values = []
                    sample_count += 1

                if buffer:
                    shard_record = self._flush_shard(
                        shards_dir=shards_dir,
                        shard_index=shard_index,
                        start_index=sample_count - len(buffer),
                        buffer=buffer,
                        source_dtypes=buffer_source_dtypes,
                    )
                    created_shard_paths.append(self.output_dir / shard_record.path)
                    shard_records.append(shard_record)
                    self._write_index_rows(
                        index_handle=index_handle,
                        shard_record=shard_record,
                        sample_source_dtypes=buffer_source_dtype_values,
                        sample_metadata_iter=metadata_iter,
                    )

                if metadata_iter is not None:
                    try:
                        next(metadata_iter)
                    except StopIteration:
                        pass
                    else:
                        raise ValueError("sample_metadata contains more rows than samples.")

            if sample_count == 0 or sample_shape is None or samples_per_shard is None:
                raise ValueError("ArrayShardStore.write requires at least one sample.")
            index_tmp_path.replace(index_path)
            created_index_path = True

            manifest = {
                "schema_version": "array_shard_store/1",
                "storage_format": "npy",
                "manifest_path": self.manifest_name,
                "index_path": self.index_name,
                "shards_dir": "shards",
                "sample_count": sample_count,
                "sample_shape": list(sample_shape),
                "source_dtypes": sorted(all_source_dtypes),
                "storage_dtype": str(self.storage_dtype),
                "target_shard_bytes": self.target_shard_bytes,
                "max_samples_per_shard": self.max_samples_per_shard,
                "samples_per_shard": samples_per_shard,
                "shard_count": len(shard_records),
                "shards": [record.to_dict() for record in shard_records],
                "extra": json_ready(dict(extra_manifest or {})),
            }
            write_json(manifest_path, manifest)
            return manifest
        except Exception:
            index_tmp_path.unlink(missing_ok=True)
            if created_index_path:
                index_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
            for tmp_path in sorted(shards_dir.glob("*.tmp")):
                tmp_path.unlink(missing_ok=True)
            for path in created_shard_paths:
                path.unlink(missing_ok=True)
            raise

    def _flush_shard(
        self,
        *,
        shards_dir: Path,
        shard_index: int,
        start_index: int,
        buffer: list[np.ndarray],
        source_dtypes: set[str],
    ) -> ShardRecord:
        shard_id = f"shard_{shard_index:05d}"
        path = shards_dir / f"{shard_id}.npy"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing shard {path}.")
        stacked = np.stack(buffer, axis=0).astype(self.storage_dtype, copy=False)
        _atomic_save_npy(path, stacked)
        return ShardRecord(
            shard_id=shard_id,
            path=str(path.relative_to(self.output_dir)),
            start_index=int(start_index),
            sample_count=int(stacked.shape[0]),
            sample_shape=tuple(int(v) for v in stacked.shape[1:]),
            source_dtypes=tuple(sorted(source_dtypes)),
            storage_dtype=str(self.storage_dtype),
            file_size_bytes=int(path.stat().st_size),
        )

    def _write_index_rows(
        self,
        *,
        index_handle: Any,
        shard_record: ShardRecord,
        sample_source_dtypes: list[str],
        sample_metadata_iter: Iterator[Mapping[str, Any]] | None,
    ) -> None:
        if len(sample_source_dtypes) != shard_record.sample_count:
            raise ValueError("sample source dtype count does not match shard sample count.")
        for offset in range(shard_record.sample_count):
            global_index = shard_record.start_index + offset
            if sample_metadata_iter is None:
                row: dict[str, Any] = {"sample_index": global_index}
            else:
                try:
                    row = dict(next(sample_metadata_iter))
                except StopIteration as exc:
                    raise ValueError("sample_metadata ended before samples.") from exc
            row.update(
                {
                    "sample_index": int(row.get("sample_index", global_index)),
                    "array_index": global_index,
                    "shard_id": shard_record.shard_id,
                    "shard_path": shard_record.path,
                    "shard_offset": offset,
                    "source_dtype": sample_source_dtypes[offset],
                    "storage_dtype": shard_record.storage_dtype,
                    "sample_shape": list(shard_record.sample_shape),
                }
            )
            index_handle.write(
                json.dumps(json_ready(row), sort_keys=True) + "\n"
            )


class ArrayShardReader:
    """Read samples from an ``ArrayShardStore`` with bounded shard caching.

    The ordinary public API is lifetime-safe: :meth:`get` and ``reader[index]``
    return independent sample copies whose lifetime does not depend on the
    backing shard remaining in the LRU cache.  Use ``get(index, copy=False)``
    only for explicit short-lived zero-copy access into the cached memmap.

    Parameters
    ----------
    root_dir:
        Directory containing the shard manifest and ``shards/`` directory.
    manifest_name:
        Name of the JSON shard manifest under ``root_dir``.
    cache_size:
        Maximum number of shard memmaps to retain at once.
    """

    root_dir: Path
    manifest: Mapping[str, Any]
    cache_size: int
    _cache: OrderedDict[str, np.ndarray]
    _shards: tuple[ShardRecord, ...]
    _starts: tuple[int, ...]

    def __init__(
        self,
        root_dir: Path,
        *,
        manifest_name: str = "array_shards_manifest.json",
        cache_size: int = 4,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.manifest = read_json(self.root_dir / manifest_name)
        self.cache_size = int(cache_size)
        if self.cache_size < 1:
            raise ValueError("cache_size must be >= 1.")
        self._cache = OrderedDict()
        self._shards = tuple(
            ShardRecord.from_dict(payload)
            for payload in self.manifest.get("shards", [])
        )
        if not self._shards:
            raise ValueError(f"No shards listed in {self.root_dir / manifest_name}.")
        self._starts = tuple(shard.start_index for shard in self._shards)
        expected_start = 0
        for shard in self._shards:
            if shard.start_index != expected_start:
                raise ValueError(
                    f"Shard {shard.shard_id} starts at {shard.start_index}, expected {expected_start}."
                )
            expected_start = shard.stop_index
        if expected_start != self.sample_count:
            raise ValueError(
                f"Shard coverage ends at {expected_start}, expected {self.sample_count} samples."
            )

    @property
    def sample_count(self) -> int:
        """Return the total sample count."""
        return int(self.manifest["sample_count"])

    @property
    def sample_shape(self) -> tuple[int, ...]:
        """Return the per-sample shape."""
        return tuple(int(v) for v in self.manifest["sample_shape"])

    @property
    def open_shard_count(self) -> int:
        """Return the current number of cached shard memory maps."""
        return len(self._cache)

    def __len__(self) -> int:
        return self.sample_count

    def __enter__(self) -> "ArrayShardReader":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close all cached memory maps."""
        for array in list(self._cache.values()):
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self._cache.clear()

    def _locate(self, index: int) -> tuple[ShardRecord, int]:
        if int(index) < 0 or int(index) >= self.sample_count:
            raise IndexError(f"sample index {index} out of range for {self.sample_count} samples.")
        shard_position = bisect_right(self._starts, int(index)) - 1
        if shard_position >= 0:
            shard = self._shards[shard_position]
            if shard.start_index <= int(index) < shard.stop_index:
                return shard, int(index) - shard.start_index
        raise ValueError(f"No shard mapping found for sample index {index}.")

    def _load_shard(self, shard: ShardRecord) -> np.ndarray:
        cached = self._cache.get(shard.shard_id)
        if cached is not None:
            self._cache.move_to_end(shard.shard_id)
            return cached
        path = self.root_dir / shard.path
        if not path.exists():
            raise FileNotFoundError(f"Missing array shard {path}.")
        try:
            array = np.load(path, mmap_mode="r")
        except Exception as exc:
            raise ValueError(f"Could not load array shard {path}: {exc}") from exc
        expected_shape = (shard.sample_count, *shard.sample_shape)
        if tuple(int(v) for v in array.shape) != expected_shape:
            raise ValueError(
                f"Shard {path} has shape {array.shape}, expected {expected_shape}."
            )
        if str(array.dtype) != shard.storage_dtype:
            raise ValueError(
                f"Shard {path} has dtype {array.dtype}, expected {shard.storage_dtype}."
            )
        self._cache[shard.shard_id] = array
        self._cache.move_to_end(shard.shard_id)
        while len(self._cache) > self.cache_size:
            _, old = self._cache.popitem(last=False)
            mmap = getattr(old, "_mmap", None)
            if mmap is not None:
                mmap.close()
        return array

    def get(self, index: int, *, copy: bool = True) -> np.ndarray:
        """Return one sample by global index.

        Parameters
        ----------
        index:
            Global sample index in the shard manifest.
        copy:
            When ``True`` (default), return an independent array.  When
            ``False``, return a view into the cached shard memmap; callers must
            consume it before LRU eviction or :meth:`close`.
        """
        shard, offset = self._locate(index)
        array = self._load_shard(shard)
        return np.array(array[offset], copy=True) if copy else np.asarray(array[offset])

    def __getitem__(self, index: int) -> np.ndarray:
        return self.get(index)
