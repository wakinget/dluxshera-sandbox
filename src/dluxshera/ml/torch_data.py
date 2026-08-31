from __future__ import annotations

from typing import Any, Mapping

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in no-torch envs
    raise ModuleNotFoundError(
        "dluxshera.ml.torch_data requires PyTorch. Install the optional ML "
        "environment, for example `python -m pip install -e .[ml]`."
    ) from exc

from .catalog import SampleCatalog
from .noise import NoiseConfig, apply_pair_noise
from .pairs import PairManifest, PairRecord, PairSampler
from .scaling import IntensityScaler

__all__ = ["DynamicPairDataset", "PairManifestDataset"]


def _record_to_tensors(
    *,
    catalog: SampleCatalog,
    reader: Any,
    record: PairRecord,
    scaler: IntensityScaler,
    noise_config: NoiseConfig,
    dynamic_seed_offset: int,
) -> dict[str, Any]:
    image_a = reader.get(int(record.sample_a_index))
    image_b = reader.get(int(record.sample_b_index))
    image_a, image_b = apply_pair_noise(
        image_a,
        image_b,
        noise_config,
        pair_record_id=record.pair_record_id,
        dynamic_seed_offset=dynamic_seed_offset,
    )
    image_a = scaler.transform(image_a)
    image_b = scaler.transform(image_b)
    return {
        "image_a": torch.from_numpy(np.asarray(image_a, dtype=np.float32)).unsqueeze(0),
        "image_b": torch.from_numpy(np.asarray(image_b, dtype=np.float32)).unsqueeze(0),
        "target_delta_z": torch.from_numpy(
            np.asarray(record.target_delta_z, dtype=np.float32)
        ),
        "target_delta_theta": torch.from_numpy(
            np.asarray(record.target_delta_theta, dtype=np.float32)
        ),
        "pair_record_id": record.pair_record_id,
        "sample_a_id": record.sample_a_id,
        "sample_b_id": record.sample_b_id,
        "eval_slice": record.eval_slice or "",
        "pair_family": record.pair_family,
        "fisher_distance_l2": torch.tensor(record.fisher_distance_l2, dtype=torch.float32),
    }


class PairManifestDataset(Dataset):
    """Read fixed ordered pairs from a frozen pair manifest."""

    def __init__(
        self,
        *,
        catalog: SampleCatalog,
        pair_manifest: PairManifest,
        scaler: IntensityScaler | None = None,
        noise_config: NoiseConfig | Mapping[str, Any] | None = None,
        shard_cache_size: int = 4,
    ) -> None:
        self.catalog = catalog
        self.records = tuple(pair_manifest.records)
        self.scaler = IntensityScaler() if scaler is None else scaler
        self.noise_config = (
            noise_config
            if isinstance(noise_config, NoiseConfig)
            else NoiseConfig.from_dict(noise_config)
        )
        self.shard_cache_size = int(shard_cache_size)
        self._reader = None

    def __len__(self) -> int:
        return len(self.records)

    def _get_reader(self) -> Any:
        if self._reader is None:
            self._reader = self.catalog.image_reader(cache_size=self.shard_cache_size)
        return self._reader

    def __getitem__(self, index: int) -> dict[str, Any]:
        return _record_to_tensors(
            catalog=self.catalog,
            reader=self._get_reader(),
            record=self.records[int(index)],
            scaler=self.scaler,
            noise_config=self.noise_config,
            dynamic_seed_offset=int(index),
        )


class DynamicPairDataset(Dataset):
    """Generate deterministic dynamic training pairs for a seed and epoch."""

    def __init__(
        self,
        *,
        catalog: SampleCatalog,
        sampler: PairSampler,
        pairs_per_epoch: int,
        seed: int,
        science_split: str = "train",
        nuisance_split: str = "train",
        scaler: IntensityScaler | None = None,
        noise_config: NoiseConfig | Mapping[str, Any] | None = None,
        shard_cache_size: int = 4,
    ) -> None:
        if int(pairs_per_epoch) < 1:
            raise ValueError("pairs_per_epoch must be >= 1.")
        self.catalog = catalog
        self.sampler = sampler
        self.pairs_per_epoch = int(pairs_per_epoch)
        self.seed = int(seed)
        self.science_split = str(science_split)
        self.nuisance_split = str(nuisance_split)
        self.scaler = IntensityScaler() if scaler is None else scaler
        self.noise_config = (
            noise_config
            if isinstance(noise_config, NoiseConfig)
            else NoiseConfig.from_dict(noise_config)
        )
        self.shard_cache_size = int(shard_cache_size)
        self.epoch = 0
        self._reader = None

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic epoch offset used by pair sampling."""
        self.epoch = int(epoch)

    def _get_reader(self) -> Any:
        if self._reader is None:
            self._reader = self.catalog.image_reader(cache_size=self.shard_cache_size)
        return self._reader

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair_seed = self.seed + self.epoch * 1_000_003 + int(index)
        rng = np.random.default_rng(pair_seed)
        record = self.sampler.sample_pair(
            rng,
            science_split=self.science_split,
            nuisance_split=self.nuisance_split,
            split="train",
            eval_slice=None,
        )
        return _record_to_tensors(
            catalog=self.catalog,
            reader=self._get_reader(),
            record=record,
            scaler=self.scaler,
            noise_config=self.noise_config,
            dynamic_seed_offset=pair_seed if self.noise_config.training_dynamic else 0,
        )
