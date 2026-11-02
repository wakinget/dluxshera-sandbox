from __future__ import annotations

import multiprocessing as mp
import os
import warnings
from typing import Any, Iterable, Mapping

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
from .noise import (
    NoiseConfig,
    apply_pair_noise,
    apply_pair_noise_batch,
    apply_pair_noise_numpy,
)
from .pairs import PairManifest, PairRecord, PairSampler, make_reverse_pair_record
from .scaling import IntensityScaler

__all__ = [
    "DynamicPairDataset",
    "JAX_BACKED_NOISE_MODELS",
    "NOISE_IMPLEMENTATIONS",
    "NOISE_IMPLEMENTATION_ENV_VAR",
    "PairManifestDataset",
    "SAFE_WORKER_START_METHODS",
    "UnsafeWorkerStartMethodError",
    "VALIDATION_NOISE_IMPLEMENTATIONS",
    "noise_requires_safe_workers",
    "resolve_dataloader_context",
    "resolve_noise_implementation",
    "resolve_validation_noise_implementation",
]

#: Noise models whose implementation executes JAX operations inside
#: ``Dataset.__getitem__``.  JAX initialises a multithreaded backend at import
#: time, and a forked child that inherits that backend deadlocks on its first
#: JAX call, so these models must not run in ``fork``-started workers.
JAX_BACKED_NOISE_MODELS = frozenset({"shera_observation"})

#: Worker start methods that re-exec a clean interpreter instead of inheriting
#: the parent's initialised JAX backend.
SAFE_WORKER_START_METHODS = ("forkserver", "spawn")

#: Preferred safe start method.  ``forkserver`` pays the interpreter startup
#: cost once per worker pool rather than once per worker, so it is cheaper than
#: ``spawn`` while being equally fork-safe.
DEFAULT_SAFE_WORKER_START_METHOD = "forkserver"

#: Supported noise *execution* strategies.  This is runtime provenance, not a
#: scientific noise identity: ``reference`` and ``batched`` produce bitwise
#: identical realizations, and ``numpy`` produces a different realization of
#: the same prescribed photon-noise probability model.
NOISE_IMPLEMENTATIONS = ("reference", "batched", "numpy")

#: Strategies permitted for *frozen* validation noise.  The S12 frozen
#: noisy-validation recipe was constructed around the JAX noise stream, so the
#: validation realization must stay JAX-backed even when training uses NumPy.
#: ``numpy`` is deliberately absent; see
#: :func:`resolve_validation_noise_implementation`.
VALIDATION_NOISE_IMPLEMENTATIONS = ("reference", "batched")

#: Environment variable supplying the runtime default noise implementation.
NOISE_IMPLEMENTATION_ENV_VAR = "DLUXSHERA_NOISE_IMPLEMENTATION"


def resolve_noise_implementation(noise_implementation: str | None = None) -> str:
    """Resolve the noise execution strategy without constructing a dataset.

    Parameters
    ----------
    noise_implementation:
        Explicit selection, or ``None`` to read
        :data:`NOISE_IMPLEMENTATION_ENV_VAR` and fall back to ``"reference"``.

    Returns
    -------
    str
        One of :data:`NOISE_IMPLEMENTATIONS`.

    Raises
    ------
    ValueError
        When the resolved value is not a supported implementation.

    Notes
    -----
    Callers that must report the resolved strategy on a path where no dataset
    exists -- for example a skipped preflight smoke test -- use this instead of
    reading ``dataset.noise_implementation``.
    """
    resolved = str(
        noise_implementation
        or os.environ.get(NOISE_IMPLEMENTATION_ENV_VAR)
        or "reference"
    )
    if resolved not in NOISE_IMPLEMENTATIONS:
        raise ValueError(
            f"noise_implementation must be one of {NOISE_IMPLEMENTATIONS}, got {resolved!r}."
        )
    return resolved


def resolve_validation_noise_implementation(training_implementation: str) -> str:
    """Return the strategy to use for frozen validation noise.

    Parameters
    ----------
    training_implementation:
        The resolved *training* noise strategy.

    Returns
    -------
    str
        One of :data:`VALIDATION_NOISE_IMPLEMENTATIONS`.

    Notes
    -----
    Selecting ``numpy`` for training maps validation to ``"reference"``.  The
    frozen S12 noisy-validation recipe pins a deterministic JAX realization of
    the validation observation images; reproducing that realization requires
    the JAX stream, so the validation loader keeps it regardless of the
    training strategy.  ``reference`` and ``batched`` are bitwise identical, so
    either preserves the frozen recipe and ``batched`` is passed through.

    This is why the training optimization can be evaluated without
    regenerating any frozen artifact.
    """
    return (
        str(training_implementation)
        if training_implementation in VALIDATION_NOISE_IMPLEMENTATIONS
        else "reference"
    )


def noise_requires_safe_workers(
    noise_configs: Iterable[NoiseConfig | Mapping[str, Any] | None]
    | NoiseConfig
    | Mapping[str, Any]
    | None,
) -> bool:
    """Return ``True`` when any enabled noise config runs JAX in dataset workers.

    Parameters
    ----------
    noise_configs:
        One noise configuration, or an iterable of them.  ``None`` entries and
        disabled configurations are ignored.

    Returns
    -------
    bool
        ``True`` when at least one enabled configuration uses a noise model
        listed in :data:`JAX_BACKED_NOISE_MODELS`.

    Notes
    -----
    Disabled noise and the ``legacy_numpy`` model never touch JAX, so noiseless
    studies keep PyTorch's default worker start method.
    """
    if noise_configs is None:
        return False
    if isinstance(noise_configs, (NoiseConfig, Mapping)):
        candidates: Iterable[Any] = (noise_configs,)
    else:
        candidates = noise_configs
    for candidate in candidates:
        if candidate is None:
            continue
        cfg = candidate if isinstance(candidate, NoiseConfig) else NoiseConfig.from_dict(candidate)
        if cfg.enabled and str(cfg.noise_model) in JAX_BACKED_NOISE_MODELS:
            return True
    return False


class UnsafeWorkerStartMethodError(RuntimeError):
    """Raised when a start method would deadlock JAX-backed dataset workers.

    This fails closed: an unsafe configuration is rejected at setup instead of
    being accepted with a warning and then stalling a multi-day allocation.
    """


def resolve_dataloader_context(
    method: str | None,
    *,
    requires_safe_workers: bool,
    num_workers: int,
    allow_unsafe_fork_deadlock: bool = False,
) -> tuple[Any, str | None, str]:
    """Resolve the multiprocessing context for a noise-aware ``DataLoader``.

    Parameters
    ----------
    method:
        Explicitly requested start method, or ``None`` to select automatically.
    requires_safe_workers:
        ``True`` when dataset items execute JAX operations.  Use
        :func:`noise_requires_safe_workers` to derive this from noise configs.
    num_workers:
        Planned ``DataLoader`` worker count.  ``0`` keeps everything in the
        training process and needs no context.
    allow_unsafe_fork_deadlock:
        Deliberately permit a known-deadlocking ``fork`` configuration.  This
        exists only so a diagnostic can *demonstrate* the deadlock.  It is not
        reachable from study configuration and must never be set by a training
        or preflight path.

    Returns
    -------
    tuple
        ``(context, resolved_name, reason)``.  ``context`` is a multiprocessing
        context object suitable for ``DataLoader(multiprocessing_context=...)``
        or ``None`` to accept the PyTorch default.  ``reason`` is a short
        human-readable provenance string for run manifests and logs.

    Raises
    ------
    UnsafeWorkerStartMethodError
        When ``fork`` is requested with ``requires_safe_workers=True`` and
        ``num_workers > 0``, unless ``allow_unsafe_fork_deadlock`` is set.
    ValueError
        When the requested start method does not exist on this platform.

    Notes
    -----
    A ``forkserver`` context is returned with its preload list cleared.  The
    default forkserver preload is ``["__main__"]``; re-importing a JAX-importing
    entry point inside the forkserver process would reinstate exactly the
    deadlock this function exists to avoid.
    """
    requested = None if method in (None, "") else str(method)
    if int(num_workers) <= 0:
        return None, None, "num_workers=0 runs dataset code in the training process"
    if requested is None:
        if not requires_safe_workers:
            return None, None, "noise does not use JAX; PyTorch default start method"
        resolved = DEFAULT_SAFE_WORKER_START_METHOD
        reason = "JAX-backed noise requires a fork-safe worker start method"
    else:
        resolved = requested
        reason = f"explicitly requested multiprocessing context {requested!r}"
        if resolved == "fork" and requires_safe_workers:
            if not allow_unsafe_fork_deadlock:
                raise UnsafeWorkerStartMethodError(
                    "Refusing to start DataLoader workers with the 'fork' start "
                    "method while JAX-backed noise is enabled. Forked workers "
                    "inherit an initialised JAX backend and deadlock on their "
                    "first JAX call, producing no error and no progress until "
                    "the job hits its wall-clock limit. Use one of "
                    f"{SAFE_WORKER_START_METHODS}, or num_workers=0. To "
                    "reproduce the deadlock deliberately, use the data-loader "
                    "diagnostic's explicit unsafe-reproduction flag."
                )
            warnings.warn(
                "Deliberately using the 'fork' start method with JAX-backed "
                "noise. This configuration is expected to deadlock and must "
                "only be used to reproduce the failure under a supervised "
                "diagnostic with a timeout.",
                RuntimeWarning,
                stacklevel=2,
            )
            reason = (
                "UNSAFE fork start method explicitly permitted for deadlock "
                "reproduction; expected to hang"
            )
    available = mp.get_all_start_methods()
    if resolved not in available:
        raise ValueError(
            f"Multiprocessing start method {resolved!r} is unavailable on this "
            f"platform; available methods are {tuple(available)}."
        )
    context = mp.get_context(resolved)
    if resolved == "forkserver":
        # Clear the default ["__main__"] preload so the forkserver process does
        # not import a JAX-initialising entry point before fork.
        context.set_forkserver_preload([])
    return context, resolved, reason


def _record_to_tensors(
    *,
    catalog: SampleCatalog,
    reader: Any,
    record: PairRecord,
    scaler: IntensityScaler,
    noise_config: NoiseConfig,
    dynamic_seed_offset: int,
    second_noise_view: bool = False,
    prepared_views: list | None = None,
) -> dict[str, Any]:
    clean_a = reader.get(int(record.sample_a_index)) if prepared_views is None else None
    clean_b = reader.get(int(record.sample_b_index)) if prepared_views is None else None
    # Noise is applied after ordering, so observation-only noise remains tied
    # to the B role even when a reverse PairRecord swaps physical samples.
    image_a, image_b = prepared_views[0] if prepared_views is not None else apply_pair_noise(
        clean_a,
        clean_b,
        noise_config,
        pair_record_id=record.pair_record_id,
        dynamic_seed_offset=dynamic_seed_offset,
    )
    image_a = scaler.transform(image_a)
    image_b = scaler.transform(image_b)
    item = {
        "image_a": torch.from_numpy(np.asarray(image_a, dtype=np.float32)).unsqueeze(0),
        "image_b": torch.from_numpy(np.asarray(image_b, dtype=np.float32)).unsqueeze(0),
        "target_delta_z": torch.from_numpy(
            np.asarray(record.target_delta_z, dtype=np.float32)
        ),
        "target_delta_theta": torch.from_numpy(
            np.asarray(record.target_delta_theta, dtype=np.float32)
        ),
        "nuisance_delta": torch.from_numpy(np.asarray(record.nuisance_delta, dtype=np.float32)),
        "pair_record_id": record.pair_record_id,
        "sample_a_id": record.sample_a_id,
        "sample_b_id": record.sample_b_id,
        "eval_slice": record.eval_slice or "",
        "pair_family": record.pair_family,
        "dataset_family": record.dataset_family_a
        if record.dataset_family_a == record.dataset_family_b
        else f"{record.dataset_family_a}->{record.dataset_family_b}",
        "distance_bin_label": record.distance_bin_label or "",
        "fisher_distance_l2": torch.tensor(record.fisher_distance_l2, dtype=torch.float32),
        "changed_science_dimensions": torch.tensor(
            record.changed_science_dimensions,
            dtype=torch.int64,
        ),
    }
    if second_noise_view:
        image_a2, image_b2 = prepared_views[1] if prepared_views is not None else apply_pair_noise(
            clean_a,
            clean_b,
            noise_config,
            pair_record_id=record.pair_record_id,
            dynamic_seed_offset=dynamic_seed_offset + 1_000_000_007,
        )
        item["image_a_noise_view2"] = torch.from_numpy(
            np.asarray(scaler.transform(image_a2), dtype=np.float32)
        ).unsqueeze(0)
        item["image_b_noise_view2"] = torch.from_numpy(
            np.asarray(scaler.transform(image_b2), dtype=np.float32)
        ).unsqueeze(0)
    return item


class _BatchFetch:
    def _single_item_needs_batch_path(self) -> bool:
        """Return ``True`` when ``__getitem__`` must delegate to ``__getitems__``.

        ``reference`` and ``batched`` are bitwise identical, so a single-item
        fetch can use the reference sampler either way.  ``numpy`` is a
        genuinely different stream, so routing one item through the reference
        sampler would make the realization depend on whether the loader chose
        the batch-fetch path.  Delegating keeps both paths identical.

        The ``enabled`` term also breaks the recursion: with noise disabled,
        ``__getitems__`` falls back to per-item fetches.
        """
        return self.noise_implementation == "numpy" and self.noise_config.enabled

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        """Fetch ordered count-space views before scaling, inside one worker."""
        if self.noise_implementation == "reference" or not self.noise_config.enabled:
            return [self[index] for index in indices]
        records_offsets = [self._record_and_offset(int(index)) for index in indices]
        reader = self._get_reader()
        # Bound all temporary storage to this batch; preserve requested order.
        clean = [(reader.get(int(r.sample_a_index)), reader.get(int(r.sample_b_index)))
                 for r, _ in records_offsets]
        # NumPy draws per record from its own derived seed, so the realization
        # is independent of this batch's width and composition.
        sampler = (apply_pair_noise_numpy if self.noise_implementation == "numpy"
                   else apply_pair_noise_batch)
        views = [sampler(clean, self.noise_config,
                 [(r.pair_record_id, offset + extra) for r, offset in records_offsets])
                 for extra in ([0, 1_000_000_007] if self.second_noise_view else [0])]
        return [_record_to_tensors(catalog=self.catalog, reader=reader, record=r,
                    scaler=self.scaler, noise_config=self.noise_config,
                    dynamic_seed_offset=offset, second_noise_view=self.second_noise_view,
                    prepared_views=[view[i] for view in views])
                for i, (r, offset) in enumerate(records_offsets)]


class PairManifestDataset(_BatchFetch, Dataset):
    """Read fixed ordered pairs from a frozen pair manifest.

    This is an optional PyTorch adapter around the architecture-neutral catalog,
    shard reader, and pair-manifest records.  Use it when a training or
    evaluation loop should replay exactly the same ordered pairs every run.

    Parameters
    ----------
    catalog:
        Prepared sample catalog used to read images.
    pair_manifest:
        Frozen ordered-pair manifest.
    scaler:
        Optional image intensity scaler.  Defaults to raw image copies.
    noise_config:
        Optional noise/augmentation configuration.  A mapping is converted with
        ``NoiseConfig.from_dict``.
    second_noise_view:
        If ``True``, include ``image_a_noise_view2`` and
        ``image_b_noise_view2`` tensors generated by reapplying
        ``noise_config`` with a separate deterministic seed offset.
    noise_implementation:
        Runtime strategy: reference (default), batched, or numpy. When omitted,
        reads DLUXSHERA_NOISE_IMPLEMENTATION. This is an execution strategy, not
        a scientific noise identity: batched is bitwise identical to reference,
        while numpy samples the same photon-noise probability model from a
        different RNG stream.
    shard_cache_size:
        Number of shard memory maps retained by the underlying image reader.

    Notes
    -----
    Items contain PyTorch tensors for ``image_a``, ``image_b``,
    ``target_delta_z``, ``target_delta_theta``, and ``nuisance_delta``, plus
    string metadata such as sample ids, pair family, and evaluation slice.  When
    ``second_noise_view=True``, items also contain ``image_a_noise_view2`` and
    ``image_b_noise_view2``.
    """

    def __init__(
        self,
        *,
        catalog: SampleCatalog,
        pair_manifest: PairManifest,
        scaler: IntensityScaler | None = None,
        noise_config: NoiseConfig | Mapping[str, Any] | None = None,
        second_noise_view: bool = False,
        shard_cache_size: int = 4,
        noise_implementation: str | None = None,
    ) -> None:
        self.noise_implementation = resolve_noise_implementation(noise_implementation)
        self.catalog = catalog
        self.records = tuple(pair_manifest.records)
        self.scaler = IntensityScaler() if scaler is None else scaler
        self.noise_config = (
            noise_config
            if isinstance(noise_config, NoiseConfig)
            else NoiseConfig.from_dict(noise_config)
        )
        self.second_noise_view = bool(second_noise_view)
        self.shard_cache_size = int(shard_cache_size)
        self._reader = None

    def __len__(self) -> int:
        return len(self.records)

    def __getstate__(self) -> dict[str, Any]:
        # Shard caches hold numpy memory maps, which pickle by value.  Drop the
        # reader so non-fork worker start methods do not ship cached shard
        # contents to every worker; each worker reopens its own memory maps.
        state = dict(self.__dict__)
        state["_reader"] = None
        return state

    def _get_reader(self) -> Any:
        if self._reader is None:
            self._reader = self.catalog.image_reader(cache_size=self.shard_cache_size)
        return self._reader

    def _record_and_offset(self, index: int) -> tuple[PairRecord, int]:
        return self.records[int(index)], int(index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._single_item_needs_batch_path():
            return self.__getitems__([int(index)])[0]
        return _record_to_tensors(
            catalog=self.catalog,
            reader=self._get_reader(),
            record=self.records[int(index)],
            scaler=self.scaler,
            noise_config=self.noise_config,
            dynamic_seed_offset=int(index),
            second_noise_view=self.second_noise_view,
        )


class DynamicPairDataset(_BatchFetch, Dataset):
    """Generate deterministic dynamic training pairs for a seed and epoch.

    This optional PyTorch adapter asks :class:`PairSampler` for pairs on demand
    instead of replaying a frozen manifest.  It is useful when training should
    see a deterministic but broader stream of controlled comparisons than a
    fixed validation/test manifest.

    ``pairs_per_epoch`` is the total number of ordered examples.  When the
    sampler policy has ``include_reverse=True``, adjacent indices are generated
    from one sampled base pair: ``2k`` is ``(A, B)`` and ``2k+1`` is its
    semantic reverse ``(B, A)``.

    Parameters
    ----------
    catalog:
        Prepared sample catalog used to read images.
    sampler:
        Pair sampler containing the catalog, split registry, and pair policy.
    pairs_per_epoch:
        Number of ordered examples exposed by ``__len__``.
    seed:
        Base seed for deterministic pair sampling.
    science_split:
        Science-state split used for training pair selection.
    nuisance_split:
        Nuisance-state split used for training pair selection.
    scaler:
        Optional image intensity scaler.  Defaults to raw image copies.
    noise_config:
        Optional noise/augmentation configuration.  A mapping is converted with
        ``NoiseConfig.from_dict``.
    second_noise_view:
        If ``True``, include ``image_a_noise_view2`` and
        ``image_b_noise_view2`` tensors generated by reapplying
        ``noise_config`` with a separate deterministic seed offset.
    noise_implementation:
        Runtime strategy: reference (default), batched, or numpy. When omitted,
        reads DLUXSHERA_NOISE_IMPLEMENTATION. This is an execution strategy, not
        a scientific noise identity: batched is bitwise identical to reference,
        while numpy samples the same photon-noise probability model from a
        different RNG stream.
    shard_cache_size:
        Number of shard memory maps retained by the underlying image reader.
    """

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
        second_noise_view: bool = False,
        shard_cache_size: int = 4,
        noise_implementation: str | None = None,
    ) -> None:
        if int(pairs_per_epoch) < 1:
            raise ValueError("pairs_per_epoch must be >= 1.")
        if sampler.policy.include_reverse and int(pairs_per_epoch) % 2 != 0:
            raise ValueError(
                "include_reverse=True requires pairs_per_epoch to be even because "
                "pairs_per_epoch counts ordered examples."
            )
        self.noise_implementation = resolve_noise_implementation(noise_implementation)
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
        self.second_noise_view = bool(second_noise_view)
        self.shard_cache_size = int(shard_cache_size)
        self.epoch = 0
        self._reader = None

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic epoch offset used by pair sampling.

        Notes
        -----
        Worker processes observe this value because the dataset is re-pickled
        for every epoch.  That requires ``persistent_workers=False``; with
        persistent workers the epoch would be frozen at its first value.
        """
        self.epoch = int(epoch)

    def __getstate__(self) -> dict[str, Any]:
        # See PairManifestDataset.__getstate__: memory maps pickle by value.
        state = dict(self.__dict__)
        state["_reader"] = None
        return state

    def _get_reader(self) -> Any:
        if self._reader is None:
            self._reader = self.catalog.image_reader(cache_size=self.shard_cache_size)
        return self._reader

    def _record_and_offset(self, index: int) -> tuple[PairRecord, int]:
        ordered_index = int(index)
        include_reverse = bool(self.sampler.policy.include_reverse)
        base_pair_index = ordered_index // 2 if include_reverse else ordered_index
        pair_seed = self.seed + self.epoch * 1_000_003 + base_pair_index
        rng = np.random.default_rng(pair_seed)
        record = self.sampler.sample_pair(
            rng,
            science_split=self.science_split,
            nuisance_split=self.nuisance_split,
            split="train",
            eval_slice=None,
            epoch=self.epoch,
        )
        if include_reverse and ordered_index % 2 == 1:
            record = make_reverse_pair_record(record)
        return record, pair_seed if self.noise_config.training_dynamic else 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._single_item_needs_batch_path():
            return self.__getitems__([int(index)])[0]
        record, offset = self._record_and_offset(index)
        return _record_to_tensors(
            catalog=self.catalog,
            reader=self._get_reader(),
            record=record,
            scaler=self.scaler,
            noise_config=self.noise_config,
            dynamic_seed_offset=offset,
            second_noise_view=self.second_noise_view,
        )
