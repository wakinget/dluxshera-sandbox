"""Runtime safety tests for the S12 photon-noise data pipeline.

These tests exercise real multiprocessing rather than mocking configuration
values, because the failure they guard against is a worker start-method
deadlock that only manifests with actual worker processes.

Two properties matter for keeping this suite honest:

- Images are at least ``DEADLOCK_IMAGE_SIZE`` on a side.  The original S12
  incident went undetected by CI because every multi-worker test used 8x8
  fixtures, and the JAX fork deadlock does not reproduce at that size.
- Every loader iteration is bounded by a watchdog.  A regression must fail the
  suite, not hang it.
"""
from __future__ import annotations

import pickle
import threading
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dluxshera.ml import (
    NoiseConfig,
    PairPolicy,
    PairSampler,
    generate_frozen_pair_manifest,
    generate_split_registry,
    load_sample_catalog,
)
from dluxshera.ml.scaling import IntensityScaler
from dluxshera.ml.torch_data import (
    DynamicPairDataset,
    PairManifestDataset,
    UnsafeWorkerStartMethodError,
    noise_requires_safe_workers,
    resolve_dataloader_context,
)
from dluxshera.ml.training import (
    MIN_SUSTAINED_SAMPLES,
    DataLoaderSmokeTimeout,
    build_bounded_dataloader_kwargs,
    run_supervised,
    smoke_test_dataloader,
)
from tests.ml.test_catalog_splits_pairs import _write_prepared_fixture

#: Smallest image size at which the JAX fork deadlock reproduces.  Measured
#: locally: 8x8 completes under fork, 32x32 hangs indefinitely.
DEADLOCK_IMAGE_SIZE = 32

#: Wall-clock budget for a bounded loader drain.  Generous relative to the
#: ~2.5s a healthy forkserver run needs, tight relative to a real deadlock.
WATCHDOG_SECONDS = 90.0

S12_TRAINING_NOISE = NoiseConfig(
    enabled=True,
    apply_to="observation",
    noise_model="shera_observation",
    photon_noise=True,
    read_noise=False,
    dark_current=False,
    bright_threshold=100.0,
    seed=12030,
    training_dynamic=True,
    negative_policy="clip",
)

S12_VALIDATION_NOISE = NoiseConfig(
    enabled=True,
    apply_to="observation",
    noise_model="shera_observation",
    photon_noise=True,
    read_noise=False,
    dark_current=False,
    bright_threshold=100.0,
    seed=12031,
    training_dynamic=False,
    negative_policy="clip",
)


class _StallForever(torch.utils.data.Dataset):
    """Dataset whose worker never returns an item.

    Defined at module level so it survives pickling into a ``spawn`` or
    ``forkserver`` worker.
    """

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int):
        threading.Event().wait()


class _Exploding(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        raise ValueError("synthetic worker failure")


def _supervised_ok() -> dict:
    return {"value": 7}


def _supervised_raises() -> dict:
    raise ValueError("synthetic probe failure")


def _supervised_hangs() -> dict:
    """Block forever, and spawn a child that also blocks forever.

    The nested child verifies that supervision kills the whole process tree,
    which is the case that matters: a deadlocked DataLoader worker is a
    grandchild of the supervising process.
    """
    import multiprocessing as inner_mp

    child = inner_mp.get_context("spawn").Process(target=_block_forever, daemon=False)
    child.start()
    threading.Event().wait()
    return {}


def _block_forever() -> None:  # pragma: no cover - runs in a grandchild process
    threading.Event().wait()


def _catalog_registry_sampler(tmp_path: Path, *, image_size: int = DEADLOCK_IMAGE_SIZE):
    catalog = load_sample_catalog(
        _write_prepared_fixture(tmp_path / "prepared", image_size=image_size)
    )
    registry = generate_split_registry(
        catalog,
        seed=3,
        science_fractions={"train": 1.0},
        nuisance_fractions={"train": 1.0},
    )
    policy = PairPolicy(
        family_weights={"same_nuisance_different_science": 1.0},
        same_pair_id=True,
        min_fisher_distance=0.5,
        max_fisher_distance=3.0,
        include_reverse=True,
        max_sampling_attempts=4000,
    )
    return catalog, registry, PairSampler(catalog, registry, policy)


def _dynamic_dataset(
    tmp_path: Path,
    *,
    noise_config: NoiseConfig = S12_TRAINING_NOISE,
    second_noise_view: bool = False,
    pairs_per_epoch: int = 8,
    image_size: int = DEADLOCK_IMAGE_SIZE,
) -> DynamicPairDataset:
    catalog, _, sampler = _catalog_registry_sampler(tmp_path, image_size=image_size)
    return DynamicPairDataset(
        catalog=catalog,
        sampler=sampler,
        pairs_per_epoch=pairs_per_epoch,
        seed=11,
        scaler=IntensityScaler(mode="global_max_abs", scale=200.0),
        noise_config=noise_config,
        second_noise_view=second_noise_view,
    )


def _safe_loader(dataset, *, num_workers: int, batch_size: int = 2, context=None):
    """Build a DataLoader using the runtime's resolved start method."""
    if context is None:
        context, _, _ = resolve_dataloader_context(
            None,
            requires_safe_workers=noise_requires_safe_workers(dataset.noise_config),
            num_workers=num_workers,
        )
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "persistent_workers": False,
    }
    if context is not None:
        kwargs["multiprocessing_context"] = context
    return torch.utils.data.DataLoader(dataset, **kwargs)


def _drain_to_file(loader, path) -> dict:
    # A file avoids tensor IPC lifetime problems and a full supervisor queue.
    batches = list(loader)
    with open(path, "wb") as stream:
        pickle.dump(batches, stream)
    return {"batches": len(batches)}


def _drain_bounded(loader, *, timeout: float = WATCHDOG_SECONDS) -> list[dict]:
    """Kill the complete probe tree if iteration or teardown deadlocks."""
    from functools import partial
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        path = Path(directory) / "batches.pkl"
        outcome = run_supervised(partial(_drain_to_file, loader, path),
                                 timeout_seconds=timeout, label="test-drain")
        assert outcome["status"] == "ok", outcome
        with path.open("rb") as stream:
            return pickle.load(stream)


def _pair_ids(batch) -> list[tuple[str, str]]:
    return list(
        zip(
            (str(v) for v in batch["sample_a_id"]),
            (str(v) for v in batch["sample_b_id"]),
        )
    )


# ---------------------------------------------------------------------------
# Start-method resolution
# ---------------------------------------------------------------------------
def test_jax_backed_noise_requires_safe_workers() -> None:
    assert noise_requires_safe_workers(S12_TRAINING_NOISE) is True
    assert noise_requires_safe_workers((S12_TRAINING_NOISE, S12_VALIDATION_NOISE)) is True


def test_noiseless_and_numpy_noise_do_not_require_safe_workers() -> None:
    """S01-S11 must keep PyTorch's default start method."""
    assert noise_requires_safe_workers(None) is False
    assert noise_requires_safe_workers(NoiseConfig(enabled=False)) is False
    assert (
        noise_requires_safe_workers(
            NoiseConfig(
                enabled=True,
                noise_model="legacy_numpy",
                read_noise=True,
                read_noise_sigma=0.5,
            )
        )
        is False
    )
    # A JAX-backed model that is disabled is also harmless.
    assert (
        noise_requires_safe_workers(
            NoiseConfig(enabled=False, noise_model="shera_observation")
        )
        is False
    )


def test_noiseless_study_keeps_pytorch_default_context() -> None:
    context, name, reason = resolve_dataloader_context(
        None, requires_safe_workers=False, num_workers=4
    )
    assert context is None
    assert name is None
    assert "default" in reason


def test_jax_noise_selects_forkserver_by_default() -> None:
    context, name, reason = resolve_dataloader_context(
        None, requires_safe_workers=True, num_workers=4
    )
    assert name == "forkserver"
    assert context is not None
    assert "fork-safe" in reason


def test_forkserver_preload_is_cleared() -> None:
    """A JAX-importing preload would reinstate the deadlock inside the forkserver.

    The default preload is ``["__main__"]``, and our entry points import the
    JAX-initialising chain at module level, so the preload must be emptied.
    Note this is process-global state on the forkserver singleton, not a
    per-context attribute.
    """
    import multiprocessing.forkserver as forkserver

    forkserver.set_forkserver_preload(["__main__"])
    _, name, _ = resolve_dataloader_context(
        None, requires_safe_workers=True, num_workers=2
    )
    assert name == "forkserver"
    assert forkserver._forkserver._preload_modules == []


def test_num_workers_zero_fallback_is_supported() -> None:
    context, name, reason = resolve_dataloader_context(
        None, requires_safe_workers=True, num_workers=0
    )
    assert context is None
    assert name is None
    assert "num_workers=0" in reason


def test_explicit_fork_with_jax_noise_fails_closed() -> None:
    """Unsafe startup must be refused, not merely warned about."""
    with pytest.raises(UnsafeWorkerStartMethodError) as excinfo:
        resolve_dataloader_context("fork", requires_safe_workers=True, num_workers=2)
    message = str(excinfo.value)
    assert "Refusing" in message
    assert "forkserver" in message
    assert "num_workers=0" in message


def test_fork_with_jax_noise_is_allowed_only_for_explicit_reproduction() -> None:
    """The unsafe path stays reachable, but only via an unmistakable opt-in."""
    with pytest.warns(RuntimeWarning, match="expected to deadlock"):
        context, name, reason = resolve_dataloader_context(
            "fork",
            requires_safe_workers=True,
            num_workers=2,
            allow_unsafe_fork_deadlock=True,
        )
    assert name == "fork"
    assert context is not None
    assert "UNSAFE" in reason


def test_fork_with_jax_noise_and_zero_workers_is_fine() -> None:
    """num_workers=0 never forks, so fork is not an unsafe request there."""
    context, name, _ = resolve_dataloader_context(
        "fork", requires_safe_workers=True, num_workers=0
    )
    assert context is None
    assert name is None


def test_explicit_fork_without_jax_noise_does_not_warn_or_raise() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, name, _ = resolve_dataloader_context(
            "fork", requires_safe_workers=False, num_workers=2
        )
    assert name == "fork"


def test_unavailable_start_method_fails_clearly() -> None:
    with pytest.raises(ValueError, match="unavailable on this platform"):
        resolve_dataloader_context(
            "definitely-not-a-method", requires_safe_workers=True, num_workers=2
        )


# ---------------------------------------------------------------------------
# Real multiprocessing with JAX-backed noise
# ---------------------------------------------------------------------------
def test_noisy_batches_retrieved_with_real_workers(tmp_path: Path) -> None:
    """The regression test for the S12 incident: real workers, realistic images."""
    dataset = _dynamic_dataset(tmp_path)
    batches = _drain_bounded(_safe_loader(dataset, num_workers=2))
    assert batches
    for batch in batches:
        assert batch["image_a"].shape[-2:] == (DEADLOCK_IMAGE_SIZE, DEADLOCK_IMAGE_SIZE)
        assert torch.isfinite(batch["image_a"]).all()
        assert torch.isfinite(batch["image_b"]).all()


def test_worker_and_inprocess_streams_agree_on_pair_identity(tmp_path: Path) -> None:
    dataset = _dynamic_dataset(tmp_path)
    serial = _drain_bounded(_safe_loader(dataset, num_workers=0))
    parallel = _drain_bounded(_safe_loader(dataset, num_workers=2))
    serial_ids = [pair for batch in serial for pair in _pair_ids(batch)]
    parallel_ids = [pair for batch in parallel for pair in _pair_ids(batch)]
    assert serial_ids == parallel_ids


def test_observation_noise_is_applied_and_model_side_is_clean(tmp_path: Path) -> None:
    """S12 applies photon noise to the observation (B) side only."""
    noisy = _dynamic_dataset(tmp_path / "noisy", noise_config=S12_TRAINING_NOISE)
    clean = _dynamic_dataset(tmp_path / "clean", noise_config=NoiseConfig(enabled=False))
    noisy_item, clean_item = noisy[0], clean[0]
    torch.testing.assert_close(noisy_item["image_a"], clean_item["image_a"])
    assert not torch.equal(noisy_item["image_b"], clean_item["image_b"])


# ---------------------------------------------------------------------------
# E01 vs E02 input handling and second-view generation
# ---------------------------------------------------------------------------
def test_e01_batches_have_no_second_noise_view(tmp_path: Path) -> None:
    dataset = _dynamic_dataset(tmp_path, second_noise_view=False)
    for batch in _drain_bounded(_safe_loader(dataset, num_workers=2)):
        assert "image_a_noise_view2" not in batch
        assert "image_b_noise_view2" not in batch


def test_e02_second_view_is_independent_realization(tmp_path: Path) -> None:
    dataset = _dynamic_dataset(tmp_path, second_noise_view=True)
    for batch in _drain_bounded(_safe_loader(dataset, num_workers=2)):
        assert "image_b_noise_view2" in batch
        # Observation side gets a genuinely different noise draw...
        assert not torch.equal(batch["image_b"], batch["image_b_noise_view2"])
        # ...while the noiseless model side is identical across views.
        torch.testing.assert_close(batch["image_a"], batch["image_a_noise_view2"])


def test_e02_second_view_is_deterministic_per_item(tmp_path: Path) -> None:
    dataset = _dynamic_dataset(tmp_path, second_noise_view=True)
    first = dataset[0]
    again = dataset[0]
    torch.testing.assert_close(first["image_b_noise_view2"], again["image_b_noise_view2"])


def test_e02_batches_drive_a_real_noise_consistency_objective(tmp_path: Path) -> None:
    """A GPU smoke test for E02 must exercise its actual objective.

    Science loss alone would not touch the second view at all, so it would pass
    even if the consistency term were broken or the views were identical.
    """
    from dluxshera.ml.losses import PairConsistencyConfig, pair_consistency_losses
    from dluxshera.ml.models import build_pairwise_correction_model

    dataset = _dynamic_dataset(tmp_path, second_noise_view=True, pairs_per_epoch=8)
    loader = _safe_loader(dataset, num_workers=2, batch_size=4)
    batch = _drain_bounded(loader)[0]
    config = PairConsistencyConfig.from_dict({"noise_consistency_weight": 0.05})
    model = build_pairwise_correction_model(
        dataset.catalog.science_dim,
        {
            "channels": [4, 8],
            "embedding_dim": 8,
            "encoder_hidden_dim": 8,
            "head_hidden_dim": 8,
            "adaptive_pool_shape": [2, 2],
        },
    )
    pred = model(batch["image_a"], batch["image_b"])
    losses = pair_consistency_losses(
        model,
        batch["image_a"],
        batch["image_b"],
        image_a_noise_view2=batch["image_a_noise_view2"],
        image_b_noise_view2=batch["image_b_noise_view2"],
        pred_ab=pred,
        config=config,
    )
    consistency = losses["noise_consistency"]
    assert torch.isfinite(consistency)
    # Independent noise realizations must produce a nonzero penalty, and the
    # term must be differentiable so it actually trains.
    assert float(consistency.detach()) > 0.0
    assert consistency.requires_grad


def test_consistency_objective_requires_the_second_view(tmp_path: Path) -> None:
    """Running an E02 weight against E01 batches must fail loudly."""
    from dluxshera.ml.losses import PairConsistencyConfig, pair_consistency_losses
    from dluxshera.ml.models import build_pairwise_correction_model

    dataset = _dynamic_dataset(tmp_path, second_noise_view=False, pairs_per_epoch=8)
    batch = _drain_bounded(_safe_loader(dataset, num_workers=0, batch_size=4))[0]
    model = build_pairwise_correction_model(
        dataset.catalog.science_dim,
        {
            "channels": [4, 8],
            "embedding_dim": 8,
            "encoder_hidden_dim": 8,
            "head_hidden_dim": 8,
            "adaptive_pool_shape": [2, 2],
        },
    )
    with pytest.raises(ValueError, match="requires second noisy pair views"):
        pair_consistency_losses(
            model,
            batch["image_a"],
            batch["image_b"],
            pred_ab=model(batch["image_a"], batch["image_b"]),
            config=PairConsistencyConfig.from_dict({"noise_consistency_weight": 0.05}),
        )


# ---------------------------------------------------------------------------
# Frozen noisy validation
# ---------------------------------------------------------------------------
def test_frozen_noisy_validation_is_reproducible_across_worker_counts(
    tmp_path: Path,
) -> None:
    catalog = load_sample_catalog(
        _write_prepared_fixture(tmp_path / "prepared", image_size=DEADLOCK_IMAGE_SIZE)
    )
    registry = generate_split_registry(
        catalog,
        seed=7,
        science_fractions={"train": 0.67, "validation": 0.33, "test": 0.0},
        nuisance_fractions={"train": 0.67, "validation": 0.33, "test": 0.0},
    )
    policy = PairPolicy(
        family_weights={"same_nuisance_different_science": 1.0},
        same_pair_id=True,
        min_fisher_distance=0.5,
        max_fisher_distance=3.0,
        max_sampling_attempts=4000,
    )
    manifest = generate_frozen_pair_manifest(
        catalog,
        registry,
        policy=policy,
        artifact_id="VAL-v1",
        seed=3,
        pairs_per_slice=4,
    )

    def dataset() -> PairManifestDataset:
        return PairManifestDataset(
            catalog=catalog,
            pair_manifest=manifest,
            scaler=IntensityScaler(mode="global_max_abs", scale=200.0),
            noise_config=S12_VALIDATION_NOISE,
        )

    serial = _drain_bounded(_safe_loader(dataset(), num_workers=0))
    parallel = _drain_bounded(_safe_loader(dataset(), num_workers=2))
    assert len(serial) == len(parallel)
    for lhs, rhs in zip(serial, parallel):
        torch.testing.assert_close(lhs["image_b"], rhs["image_b"])
        assert _pair_ids(lhs) == _pair_ids(rhs)


# ---------------------------------------------------------------------------
# Epoch-dependent sampling across worker boundaries
# ---------------------------------------------------------------------------
def test_epoch_state_propagates_to_workers_and_changes_pairs(tmp_path: Path) -> None:
    dataset = _dynamic_dataset(tmp_path, pairs_per_epoch=8)
    loader = _safe_loader(dataset, num_workers=2)

    dataset.set_epoch(0)
    epoch0 = [p for b in _drain_bounded(loader) for p in _pair_ids(b)]
    dataset.set_epoch(0)
    epoch0_again = [p for b in _drain_bounded(loader) for p in _pair_ids(b)]
    dataset.set_epoch(1)
    epoch1 = [p for b in _drain_bounded(loader) for p in _pair_ids(b)]

    assert epoch0 == epoch0_again
    assert epoch1 != epoch0


def test_reverse_pair_adjacency_survives_epoch_transition(tmp_path: Path) -> None:
    dataset = _dynamic_dataset(tmp_path, pairs_per_epoch=8)
    loader = _safe_loader(dataset, num_workers=2)
    for epoch in (0, 1, 2):
        dataset.set_epoch(epoch)
        pairs = [p for b in _drain_bounded(loader) for p in _pair_ids(b)]
        for index in range(0, len(pairs), 2):
            assert pairs[index + 1] == tuple(reversed(pairs[index]))


def test_noise_seeds_differ_across_epochs_for_dynamic_training_noise(
    tmp_path: Path,
) -> None:
    dataset = _dynamic_dataset(tmp_path, pairs_per_epoch=2)
    dataset.set_epoch(0)
    epoch0 = dataset[0]["image_b"].clone()
    dataset.set_epoch(1)
    epoch1 = dataset[0]["image_b"].clone()
    assert not torch.equal(epoch0, epoch1)


# ---------------------------------------------------------------------------
# Serialization under non-fork start methods
# ---------------------------------------------------------------------------
def test_dataset_pickles_and_drops_warm_shard_reader(tmp_path: Path) -> None:
    """Shard memory maps pickle by value; the reader must not ride along.

    Under forkserver/spawn the dataset is re-pickled every epoch, so a cached
    memory map would ship shard contents to every worker on every epoch.
    """
    dataset = _dynamic_dataset(tmp_path)
    _ = dataset[0]
    assert dataset._reader is not None
    assert dataset._reader.open_shard_count > 0

    blob = pickle.dumps(dataset)
    assert b"memmap" not in blob
    restored = pickle.loads(blob)
    assert restored._reader is None
    # The restored dataset still works; it reopens its own memory maps.
    assert restored[0]["image_b"].shape == dataset[0]["image_b"].shape


def test_manifest_dataset_pickles_without_reader(tmp_path: Path) -> None:
    catalog, registry, sampler = _catalog_registry_sampler(tmp_path)
    manifest = generate_frozen_pair_manifest(
        catalog,
        registry,
        policy=sampler.policy,
        artifact_id="VAL-v1",
        seed=5,
        pairs_per_slice=2,
        eval_slices={"train_seen": {"science_split": "train", "nuisance_split": "train"}},
    )
    dataset = PairManifestDataset(
        catalog=catalog,
        pair_manifest=manifest,
        noise_config=S12_VALIDATION_NOISE,
    )
    _ = dataset[0]
    assert dataset._reader is not None
    blob = pickle.dumps(dataset)
    assert b"memmap" not in blob
    assert pickle.loads(blob)._reader is None


# ---------------------------------------------------------------------------
# Fail-fast diagnostics
# ---------------------------------------------------------------------------
def test_smoke_test_reports_timing_and_runtime_identity(tmp_path: Path) -> None:
    dataset = _dynamic_dataset(tmp_path, pairs_per_epoch=8)
    result = smoke_test_dataloader(
        _safe_loader(dataset, num_workers=2),
        batches=2,
        timeout_seconds=WATCHDOG_SECONDS,
    )
    assert result["batches_observed"] == 2
    assert result["first_batch_seconds"] > 0.0
    assert result["multiprocessing_method"] == "forkserver"
    assert result["num_workers"] == 2
    assert result["jax_backend"]
    assert result["image_batch_shape"][-2:] == [
        DEADLOCK_IMAGE_SIZE,
        DEADLOCK_IMAGE_SIZE,
    ]


def test_smoke_test_refuses_to_vouch_for_undersampled_throughput(
    tmp_path: Path,
) -> None:
    """A handful of batches is not a defensible capacity estimate."""
    dataset = _dynamic_dataset(tmp_path, pairs_per_epoch=8)
    result = smoke_test_dataloader(
        _safe_loader(dataset, num_workers=2),
        batches=2,
        timeout_seconds=WATCHDOG_SECONDS,
    )
    assert result["sustained_measurement_valid"] is False
    assert "defensible capacity estimate" in result["sustained_measurement_note"]
    # Warmup defaults to the prefetch depth, so a 2-batch run measures nothing.
    assert result["warmup_batches_discarded"] >= 2
    assert result["sustained_batches"] == 0


def test_sustained_measurement_discards_prefetch_warmup(tmp_path: Path) -> None:
    """Post-warmup samples must exclude the startup-filled queue."""
    dataset = _dynamic_dataset(tmp_path, pairs_per_epoch=32)
    result = smoke_test_dataloader(
        _safe_loader(dataset, num_workers=2, batch_size=2),
        batches=12,
        timeout_seconds=WATCHDOG_SECONDS,
        warmup_batches=4,
    )
    assert result["batches_observed"] == 12
    assert result["warmup_batches_discarded"] == 4
    assert result["sustained_batches"] == 8
    assert result["sustained_pairs_per_second"] > 0.0
    # Eight fast fixture batches still fall under the time threshold, so the
    # measurement must not claim to be defensible.
    assert result["sustained_measurement_valid"] is False


def test_sustained_validity_requires_both_sample_and_time_thresholds() -> None:
    """Many fast batches alone do not make a defensible estimate."""

    class Trivial(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return MIN_SUSTAINED_SAMPLES * 4

        def __getitem__(self, index: int):
            return torch.zeros(1)

    loader = torch.utils.data.DataLoader(Trivial(), batch_size=1, num_workers=0)
    result = smoke_test_dataloader(
        loader, batches=MIN_SUSTAINED_SAMPLES * 2, timeout_seconds=60.0
    )
    assert result["sustained_batches"] >= MIN_SUSTAINED_SAMPLES
    # Sample count satisfied, elapsed time nowhere near it.
    assert result["sustained_measurement_valid"] is False
    assert "spanning" in result["sustained_measurement_note"]


def test_smoke_test_works_without_workers(tmp_path: Path) -> None:
    dataset = _dynamic_dataset(tmp_path, pairs_per_epoch=8)
    result = smoke_test_dataloader(
        _safe_loader(dataset, num_workers=0),
        batches=2,
        timeout_seconds=WATCHDOG_SECONDS,
    )
    assert result["batches_observed"] == 2
    assert result["num_workers"] == 0
    assert result["multiprocessing_method"] is None
    # No worker queue, so nothing is discarded as warmup.
    assert result["warmup_batches_discarded"] == 0


def test_smoke_test_reports_timeout_when_too_few_batches_arrive() -> None:
    """Exhausting the budget yields diagnostics, not an indefinite block."""

    class Slow(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 64

        def __getitem__(self, index: int):
            time.sleep(0.3)
            return torch.zeros(1)

    loader = torch.utils.data.DataLoader(Slow(), batch_size=1, num_workers=0)
    with pytest.raises(DataLoaderSmokeTimeout) as excinfo:
        smoke_test_dataloader(loader, batches=64, timeout_seconds=1.0)
    message = str(excinfo.value)
    assert "of 64 batch(es)" in message
    assert "deadlock" in message
    assert "throughput" in message


def test_dataloader_per_batch_timeout_covers_a_stalled_first_batch() -> None:
    """DataLoader(timeout=) does fire on the first fetch, not only later ones.

    This is the per-batch bound used in production training; it is what turns a
    wedged worker into a prompt error instead of a multi-day stall.
    """
    loader = torch.utils.data.DataLoader(
        _StallForever(),
        batch_size=1,
        num_workers=1,
        **build_bounded_dataloader_kwargs(num_workers=1, per_batch_timeout_seconds=5.0),
    )
    assert loader.timeout == 5.0
    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="timed out"):
        smoke_test_dataloader(loader, batches=1, timeout_seconds=WATCHDOG_SECONDS)
    # The per-batch timeout, not the overall budget, is what fired.
    assert time.perf_counter() - started < 30.0


def test_bounded_dataloader_kwargs_skip_timeout_without_workers() -> None:
    """PyTorch rejects timeout>0 with num_workers=0."""
    assert build_bounded_dataloader_kwargs(
        num_workers=0, per_batch_timeout_seconds=30.0
    ) == {}
    assert build_bounded_dataloader_kwargs(
        num_workers=4, per_batch_timeout_seconds=None
    ) == {}
    assert build_bounded_dataloader_kwargs(
        num_workers=4, per_batch_timeout_seconds=0.0
    ) == {}
    assert build_bounded_dataloader_kwargs(
        num_workers=4, per_batch_timeout_seconds=30.0
    ) == {"timeout": 30.0}


def test_smoke_test_surfaces_worker_failure_detail() -> None:
    loader = torch.utils.data.DataLoader(_Exploding(), batch_size=1, num_workers=0)
    with pytest.raises(RuntimeError, match="synthetic worker failure"):
        smoke_test_dataloader(loader, batches=1, timeout_seconds=30.0)


def test_smoke_test_rejects_nonpositive_timeout(tmp_path: Path) -> None:
    dataset = _dynamic_dataset(tmp_path, pairs_per_epoch=8)
    with pytest.raises(ValueError, match="timeout_seconds must be > 0"):
        smoke_test_dataloader(
            _safe_loader(dataset, num_workers=0), batches=1, timeout_seconds=0.0
        )


# ---------------------------------------------------------------------------
# Process-level supervision
# ---------------------------------------------------------------------------
def test_supervised_probe_returns_result_on_success() -> None:
    outcome = run_supervised(_supervised_ok, timeout_seconds=120.0, label="ok")
    assert outcome["status"] == "ok"
    assert outcome["result"] == {"value": 7}
    assert outcome["elapsed_seconds"] > 0.0


def test_supervised_probe_reports_child_failure() -> None:
    outcome = run_supervised(_supervised_raises, timeout_seconds=120.0, label="boom")
    assert outcome["status"] == "failed"
    assert "synthetic probe failure" in outcome["error"]


def test_dataloader_smoke_controls_survive_submitted_env_filtering(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Documented ML_DATALOADER_SMOKE* controls must reach the batch job.

    prepare_sbatch_submission strips parent-shell ML_* variables and rebuilds
    the environment from the SUBMITTED_ENV_KEYS allowlist, so a key missing from
    that tuple is silently dropped and the job quietly runs with defaults.
    """
    import importlib.util
    import json

    from dluxshera.ml.hpc import SUBMITTED_ENV_KEYS

    for key in (
        "ML_DATALOADER_SMOKE",
        "ML_DATALOADER_SMOKE_BATCHES",
        "ML_DATALOADER_SMOKE_TIMEOUT",
    ):
        assert key in SUBMITTED_ENV_KEYS

    # A stale parent-shell value must not leak through.
    monkeypatch.setenv("ML_DATALOADER_SMOKE", "0")

    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "_submit_study_run_for_test",
        repo_root / "work" / "experiments" / "ml" / "hpc" / "submit_study_run.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    rc = module.main(
        [
            "--site",
            "gattaca2",
            "--study",
            str(repo_root / "work" / "experiments" / "ml" / "s12" / "study.yaml"),
            "--experiment-id",
            "S12-E01",
            "--run-id",
            "S12-E01-R001",
            "--prepared-root",
            "/scratch/prepared/PREP-V4-v1",
            "--split-registry",
            "/scratch/artifacts/split.json",
            "--validation-manifest",
            "/scratch/artifacts/val",
            "--test-manifest",
            "/scratch/artifacts/test",
            "--run-dir",
            "/scratch/runs_s12_restart",
            "--dataloader-smoke",
            "--dataloader-smoke-batches",
            "24",
            "--dataloader-smoke-timeout",
            "900",
            "--extra-sbatch-arg=--time=3-00:00:00",
            "--dry-run",
        ]
    )
    assert rc == 0
    preview = json.loads(capsys.readouterr().out)
    assert "--time=3-00:00:00" in preview["command"]
    env = preview["environment"]
    assert env["ML_DATALOADER_SMOKE"] == "1"
    assert env["ML_DATALOADER_SMOKE_BATCHES"] == "24"
    assert env["ML_DATALOADER_SMOKE_TIMEOUT"] == "900.0"
    # Restart must target the fresh root, never overwrite the failed runs.
    assert env["ML_RUN_DIR"] == "/scratch/runs_s12_restart"
    assert "ML_OVERWRITE" not in env


def test_supervised_probe_kills_unkillable_child_on_timeout() -> None:
    """The decisive property: a hung probe is terminated, not merely abandoned.

    An in-process watchdog thread cannot do this, which is why diagnostics run
    under process-level supervision.
    """
    started = time.perf_counter()
    outcome = run_supervised(_supervised_hangs, timeout_seconds=5.0, label="hang")
    elapsed = time.perf_counter() - started
    assert outcome["status"] == "timeout"
    assert "killed" in outcome["detail"]
    assert elapsed < 60.0


def test_sustained_pairs_count_partial_final_batch() -> None:
    loader = torch.utils.data.DataLoader(
        [{"image_a": torch.zeros(1)} for _ in range(5)], batch_size=2)
    result = smoke_test_dataloader(loader, batches=3, timeout_seconds=30, warmup_batches=0)
    assert result["sustained_batches"] == 3
    assert result["sustained_pairs"] == 5
    assert result["sustained_pairs_per_second"] == pytest.approx(5 / result["sustained_seconds"])


def test_zero_worker_preflight_has_process_deadline(tmp_path) -> None:
    import sys
    from unittest.mock import patch
    hpc = Path(__file__).resolve().parents[2] / "work/experiments/ml/hpc"
    with patch.object(sys, "path", [str(hpc), *sys.path]):
        import preflight_ml_gpu as preflight
        catalog, registry, sampler = _catalog_registry_sampler(tmp_path)
        outcome = preflight._dataloader_smoke(
            config={"noise": S12_TRAINING_NOISE.to_dict(),
                    "training": {"num_workers": 0, "batch_size": 2}},
            catalog=catalog, split_registry=registry, pair_policy=sampler.policy,
            scaler_path=None, enabled=True, batches=2, timeout_seconds=90)
        assert outcome["deadline_kind"] == "process"
        assert outcome["batches_observed"] == 2


def test_s12_plan_uses_fresh_nested_paths() -> None:
    from dluxshera.ml import expand_study_run_plan, load_study_prescription
    study = load_study_prescription(Path("work/experiments/ml/s12/study.yaml"))
    rows = expand_study_run_plan(study, output_root="/scratch/fresh")
    assert len(rows) == 6
    for row in rows:
        assert row.output_root == f"/scratch/fresh/S12/{row.experiment_id}/{row.run_id}"


@pytest.mark.parametrize("budget", [0, -1, float("inf"), float("nan")])
def test_supervisor_rejects_invalid_deadline(budget) -> None:
    with pytest.raises(ValueError, match="finite and > 0"):
        run_supervised(_supervised_ok, timeout_seconds=budget)
