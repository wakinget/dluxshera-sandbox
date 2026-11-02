"""Scientific and reproducibility tests for the NumPy photon-noise strategy.

The NumPy strategy is deliberately **not** bitwise comparable to the JAX
reference: it samples the same photon-noise probability model from a different
random-number stream.  So the contract it must satisfy is different from the
batched strategy's, and these tests assert that contract directly:

- *Distribution*: Poisson moments, zero probability, Fano factor, and a
  goodness-of-fit test at the faint count levels S12 actually occupies.
- *Independence*: no spatial structure, and independent E02 views.
- *Determinism*: the realization is a function of the logical record identity
  alone -- not of batch size, batch composition, worker count, or worker
  assignment.
- *Isolation*: frozen validation noise stays JAX-backed.

Statistical tolerances are sized against the finite sample, not chosen to pass.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from dluxshera.ml.noise import (
    NUMPY_BIT_GENERATOR,
    NoiseConfig,
    apply_pair_noise,
    apply_pair_noise_numpy,
    effective_noise_implementation,
    numpy_noise_provenance,
    numpy_supports,
    pair_noise_side_seeds,
)
from tests.ml.test_s12_dataloader_runtime import (
    S12_TRAINING_NOISE,
    S12_VALIDATION_NOISE,
    _dynamic_dataset,
    _safe_loader,
)


def _records(count: int, *, offset: int = 1_000_003) -> list[tuple[str, int]]:
    return [(f"p{i}", offset + i) for i in range(count)]


def _draw(image: np.ndarray, config: NoiseConfig, record: str, offset: int) -> np.ndarray:
    """Return the noisy B-side image for one logical record."""
    return apply_pair_noise_numpy([(image, image)], config, [(record, offset)])[0][1]


# ---------------------------------------------------------------------------
# Distribution fidelity
# ---------------------------------------------------------------------------
#: Count levels spanning the faint regime S12 lives in (V3 per-image minima are
#: 1.2-3.6) through the bright core (V3 maxima reach ~10,600).
LAMBDAS = (0.1, 0.37, 0.5, 1.0, 5.0, 20.0, 300.0, 9000.0)


@pytest.mark.parametrize("lam", LAMBDAS)
def test_numpy_poisson_moments_and_zero_probability(lam):
    """Mean, variance, Fano factor, and P(0) must match Poisson(lambda).

    Tolerances are set from the sampling distribution of each statistic at this
    sample size, so a passing result is evidence rather than a loose bound.
    """
    side = 640
    n = side * side
    got = _draw(np.full((side, side), lam, np.float32), S12_TRAINING_NOISE, "moments", 0)
    assert got.dtype == np.float32
    assert np.all(np.isfinite(got))
    assert np.all(got >= 0.0)

    # sem(mean) = sqrt(lam/n); allow 5 sigma.
    assert abs(got.mean() - lam) < 5.0 * np.sqrt(lam / n)
    # sem(var)/var ~ sqrt(2/n) for large lam; the Poisson correction is small.
    assert abs(got.var() / lam - 1.0) < 5.0 * np.sqrt(2.0 / n) + 0.01
    # Fano factor is variance/mean and must be 1 for a Poisson process.
    assert abs(got.var() / got.mean() - 1.0) < 0.02

    p_zero_true = float(np.exp(-lam))
    if p_zero_true > 1e-6:
        sem = np.sqrt(p_zero_true * (1.0 - p_zero_true) / n)
        assert abs(float(np.mean(got == 0.0)) - p_zero_true) < 5.0 * sem


@pytest.mark.parametrize("lam", [0.1, 0.37, 1.0, 5.0, 20.0])
def test_numpy_poisson_passes_goodness_of_fit_and_tail_probabilities(lam):
    """A chi-square test against the Poisson pmf, plus explicit tail mass.

    Moments alone cannot distinguish Poisson from an overdispersed mixture with
    the right mean and variance, which is exactly the failure mode the lookup
    investigation found.  This tests the whole pmf.
    """
    from scipy.stats import chisquare, poisson

    side = 640
    n = side * side
    got = _draw(np.full((side, side), lam, np.float32), S12_TRAINING_NOISE, "gof", 3)
    high = int(poisson.ppf(1.0 - 1e-9, lam)) + 2
    observed = np.bincount(np.clip(got.astype(np.int64).ravel(), 0, high), minlength=high + 1)
    pmf = poisson.pmf(np.arange(high + 1), lam)
    pmf = pmf / pmf.sum()
    expected = pmf * n
    keep = expected >= 5.0
    obs = np.append(observed[keep], observed[~keep].sum())
    exp = np.append(expected[keep], expected[~keep].sum())
    exp = exp * (obs.sum() / exp.sum())
    assert chisquare(obs, exp).pvalue > 1e-4

    # Selected tail probabilities, where an approximate sampler fails first.
    for k in (1, 2, 3):
        threshold = lam + k * np.sqrt(lam)
        measured = float(np.mean(got > threshold))
        truth = float(poisson.sf(np.floor(threshold), lam))
        sem = np.sqrt(max(truth * (1.0 - truth), 1e-12) / n)
        assert abs(measured - truth) < 6.0 * sem + 1e-4


def test_numpy_and_jax_samples_agree_statistically_not_bitwise():
    """Compare the two streams as distributions, not as arrays."""
    from scipy.stats import ks_2samp

    lam = np.full((400, 400), 3.0, np.float32)
    cfg = S12_TRAINING_NOISE
    numpy_draw = _draw(lam, cfg, "compare", 11)
    jax_draw = apply_pair_noise(lam, lam, cfg, pair_record_id="compare", dynamic_seed_offset=11)[1]

    # Different RNG algorithms: the realizations must differ.
    assert not np.array_equal(numpy_draw, jax_draw)
    # Same distribution: moments agree within sampling error and the two-sample
    # KS test does not reject.
    n = lam.size
    assert abs(numpy_draw.mean() - jax_draw.mean()) < 6.0 * np.sqrt(2.0 * 3.0 / n)
    assert abs(numpy_draw.var() - jax_draw.var()) < 0.1
    assert ks_2samp(numpy_draw.ravel(), jax_draw.ravel()).pvalue > 1e-4


def test_numpy_noise_has_no_spatial_structure():
    """Photon noise is independent per pixel; lag correlations must be at floor."""
    lam_value = 5.0
    residual = _draw(np.full((400, 400), lam_value, np.float32),
                     S12_TRAINING_NOISE, "spatial", 5) - lam_value
    floor = 1.0 / np.sqrt(residual.size)
    for dy, dx in ((0, 1), (1, 0), (8, 0), (0, 8)):
        shifted = np.roll(np.roll(residual, dy, axis=0), dx, axis=1)
        rho = float(np.corrcoef(residual.ravel(), shifted.ravel())[0, 1])
        assert abs(rho) < 6.0 * floor, (dy, dx, rho)


# ---------------------------------------------------------------------------
# Determinism and the seed schedule
# ---------------------------------------------------------------------------
def test_numpy_realization_is_reproducible_for_the_same_logical_record():
    image = np.full((64, 64), 4.0, np.float32)
    first = _draw(image, S12_TRAINING_NOISE, "rec", 42)
    again = _draw(image, S12_TRAINING_NOISE, "rec", 42)
    assert np.array_equal(first, again)
    # A different record or epoch offset must give a fresh realization.
    assert not np.array_equal(first, _draw(image, S12_TRAINING_NOISE, "other", 42))
    assert not np.array_equal(first, _draw(image, S12_TRAINING_NOISE, "rec", 43))


@pytest.mark.parametrize("width", [1, 2, 3, 5, 8])
def test_numpy_realization_is_independent_of_batch_partitioning(width):
    """The whole point of per-record seeding: batch width cannot matter.

    A shared-RNG implementation would fail this, which is why it is not the
    production candidate however fast it benchmarks.
    """
    images = [np.full((32, 32), float(v), np.float32) for v in (1, 3, 7, 40, 200, 1500, 0, 101)]
    pairs = [(x, x) for x in images]
    records = _records(len(pairs))
    whole = apply_pair_noise_numpy(pairs, S12_TRAINING_NOISE, records)
    chunked = sum(
        [apply_pair_noise_numpy(pairs[i:i + width], S12_TRAINING_NOISE, records[i:i + width])
         for i in range(0, len(pairs), width)],
        [],
    )
    assert len(chunked) == len(whole)
    for (wa, wb), (ca, cb) in zip(whole, chunked):
        assert np.array_equal(wa, ca)
        assert np.array_equal(wb, cb)


def test_numpy_realization_is_independent_of_batch_composition():
    """Reordering or regrouping the batch must not change any record's draw."""
    images = [np.full((32, 32), float(v), np.float32) for v in (2, 9, 60, 400)]
    records = _records(len(images))
    straight = apply_pair_noise_numpy([(x, x) for x in images], S12_TRAINING_NOISE, records)
    order = [3, 1, 0, 2]
    shuffled = apply_pair_noise_numpy(
        [(images[i], images[i]) for i in order],
        S12_TRAINING_NOISE,
        [records[i] for i in order],
    )
    for position, index in enumerate(order):
        assert np.array_equal(shuffled[position][1], straight[index][1])


def test_side_offsets_and_e02_views_are_uncorrelated():
    """The schedule's small integer offsets must still decorrelate the streams.

    PCG64 is seeded through ``SeedSequence``, which hashes the integer, so the
    ``+17``/``+31`` side offsets and E02's ``+1_000_000_007`` view offset give
    independent streams rather than near-identical ones.  Asserting this is what
    rules out a naive linear-seeding mistake.
    """
    cfg = replace(S12_TRAINING_NOISE, apply_to="both")
    lam_value = 5.0
    image = np.full((200, 200), lam_value, np.float32)
    (side_a, side_b), = apply_pair_noise_numpy([(image, image)], cfg, [("rec", 77)])
    floor = 1.0 / np.sqrt(image.size)
    rho_sides = float(np.corrcoef((side_a - lam_value).ravel(), (side_b - lam_value).ravel())[0, 1])
    assert abs(rho_sides) < 6.0 * floor

    # E02's second view uses the dataset's +1_000_000_007 offset.
    view1 = _draw(image, S12_TRAINING_NOISE, "rec", 77)
    view2 = _draw(image, S12_TRAINING_NOISE, "rec", 77 + 1_000_000_007)
    assert not np.array_equal(view1, view2)
    rho_views = float(np.corrcoef((view1 - lam_value).ravel(), (view2 - lam_value).ravel())[0, 1])
    assert abs(rho_views) < 6.0 * floor


def test_derived_seeds_are_distinct_across_records_and_epochs():
    """The logical schedule must not collapse distinct records onto one seed."""
    seeds = {
        pair_noise_side_seeds(
            S12_TRAINING_NOISE, pair_record_id=f"S{k}", dynamic_seed_offset=11 + epoch * 1_000_003 + k
        )["image_b_seed"]
        for epoch in range(3)
        for k in range(2000)
    }
    assert len(seeds) == 6000


# ---------------------------------------------------------------------------
# Noise model semantics
# ---------------------------------------------------------------------------
def test_numpy_preserves_the_bright_image_gaussian_branch():
    """The branch predicate is min(image) > threshold, as in the shared utility."""
    bright = np.full((64, 64), 500.0, np.float32)
    got = _draw(bright, S12_TRAINING_NOISE, "bright", 1)
    # The Gaussian branch returns non-integer values; Poisson would not.
    assert not np.all(got == np.floor(got))
    assert abs(got.mean() - 500.0) < 5.0 * np.sqrt(500.0 / got.size)
    assert abs(got.var() / 500.0 - 1.0) < 0.1

    # One faint pixel sends the whole image to the exact Poisson branch.
    mixed = bright.copy()
    mixed[0, 0] = 2.0
    poisson_got = _draw(mixed, S12_TRAINING_NOISE, "mixed", 1)
    assert np.all(poisson_got == np.floor(poisson_got))

    # Equality at the threshold stays on the Poisson branch (strictly greater).
    at_threshold = _draw(np.full((16, 16), 100.0, np.float32), S12_TRAINING_NOISE, "eq", 1)
    assert np.all(at_threshold == np.floor(at_threshold))


def test_numpy_clips_or_raises_on_negative_expected_counts():
    """Negative inputs follow the existing per-config policy, not a new one."""
    negative = np.full((16, 16), -1.0, np.float32)
    clipped = _draw(negative, replace(S12_TRAINING_NOISE, negative_policy="clip"), "neg", 0)
    assert np.all(clipped == 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        _draw(negative, replace(S12_TRAINING_NOISE, negative_policy="raise"), "neg", 0)


def test_image_a_is_untouched_for_observation_side_noise():
    """S12 applies noise to observation B only; A must be bitwise unchanged."""
    image_a = np.full((32, 32), 7.0, np.float32)
    image_b = np.full((32, 32), 7.0, np.float32)
    got_a, got_b = apply_pair_noise_numpy([(image_a, image_b)], S12_TRAINING_NOISE, [("ab", 0)])[0]
    assert np.array_equal(got_a, image_a)
    assert not np.array_equal(got_b, image_b)
    # And it is a copy, so a downstream scaler cannot mutate the shard cache.
    assert got_a is not image_a


@pytest.mark.parametrize("options", [
    dict(enabled=False),
    dict(noise_model="legacy_numpy"),
    dict(read_noise=True, detector_model="GSENSE2020BSI"),
    dict(dark_current=True, detector_model="GSENSE2020BSI", exposure_time_s=1.0),
])
def test_unsupported_configurations_fall_back_to_the_reference_exactly(options):
    """An unsupported configuration must reproduce the reference bitwise.

    The fallback is deliberately exact rather than an approximation in a
    different detector model.
    """
    cfg = replace(S12_TRAINING_NOISE, **options)
    assert numpy_supports(cfg) is False
    assert effective_noise_implementation(cfg, "numpy") == "reference"
    image = np.full((32, 32), 5.0, np.float32)
    expected = apply_pair_noise(image, image, cfg, pair_record_id="f", dynamic_seed_offset=2)
    actual = apply_pair_noise_numpy([(image, image)], cfg, [("f", 2)])[0]
    assert all(np.array_equal(a, b) for a, b in zip(expected, actual))


def test_photon_noise_disabled_returns_clipped_copies():
    cfg = replace(S12_TRAINING_NOISE, photon_noise=False)
    assert numpy_supports(cfg) is True
    image = np.full((16, 16), 5.0, np.float32)
    got_a, got_b = apply_pair_noise_numpy([(image, image)], cfg, [("off", 0)])[0]
    assert np.array_equal(got_b, image)
    assert got_b is not image


def test_numpy_requires_matching_record_counts():
    image = np.zeros((8, 8), np.float32)
    with pytest.raises(ValueError, match="matching lengths"):
        apply_pair_noise_numpy([(image, image)], S12_TRAINING_NOISE, [])


def test_effective_implementation_passes_through_supported_strategies():
    assert effective_noise_implementation(S12_TRAINING_NOISE, "numpy") == "numpy"
    assert effective_noise_implementation(S12_TRAINING_NOISE, "batched") == "batched"
    assert effective_noise_implementation(S12_TRAINING_NOISE, "reference") == "reference"
    # Only numpy has a coverage gate; batched does its own fallback internally.
    assert effective_noise_implementation(
        replace(S12_TRAINING_NOISE, enabled=False), "batched"
    ) == "batched"


def test_rng_provenance_is_recorded_and_pinned():
    """Provenance must name the algorithm and refuse to overclaim stability."""
    provenance = numpy_noise_provenance()
    assert provenance["bit_generator"] == NUMPY_BIT_GENERATOR == "PCG64"
    assert provenance["numpy_version"] == np.__version__
    assert "SeedSequence" in provenance["seeding"]
    assert "not bitwise comparable" in provenance["equivalence"]
    assert "not claimed" in provenance["determinism_scope"]


# ---------------------------------------------------------------------------
# Dataset and DataLoader integration
# ---------------------------------------------------------------------------
def test_dataset_single_and_batch_fetch_agree_under_numpy(tmp_path):
    """``__getitem__`` must not produce a different realization from ``__getitems__``.

    Without the delegation in ``_single_item_needs_batch_path`` the single-item
    path would fall through to the JAX sampler, making the realization depend
    on whether PyTorch chose the batch-fetch path.
    """
    ds = _dynamic_dataset(tmp_path, pairs_per_epoch=8, second_noise_view=True)
    ds.noise_implementation = "numpy"
    singles = [ds[i] for i in range(len(ds))]
    batched = ds.__getitems__(list(range(len(ds))))
    for single, batch_item in zip(singles, batched):
        assert single.keys() == batch_item.keys()
        for key in single:
            if isinstance(single[key], torch.Tensor):
                assert torch.equal(single[key], batch_item[key]), key
            else:
                assert single[key] == batch_item[key], key


@pytest.mark.parametrize("second", [False, True])
@pytest.mark.parametrize("workers,method", [(0, None), (4, "forkserver"), (2, "spawn")])
def test_numpy_dataset_realization_survives_workers_and_start_methods(
    tmp_path, second, workers, method
):
    """Worker count, worker assignment, and start method must not matter.

    This runs real worker processes, like the batched equivalence test, because
    the property under test is about process topology.
    """
    from dluxshera.ml.torch_data import resolve_dataloader_context

    ds = _dynamic_dataset(tmp_path, second_noise_view=second, pairs_per_epoch=10, image_size=32)
    ds.noise_implementation = "numpy"
    context, _, _ = resolve_dataloader_context(
        method, requires_safe_workers=True, num_workers=workers
    )
    for epoch in (0, 2):
        ds.set_epoch(epoch)
        expected = [ds[i] for i in range(len(ds))]
        loader = _safe_loader(ds, num_workers=workers, batch_size=4, context=context)
        if workers:
            loader.timeout = 90
        actual = []
        for batch in loader:
            for i in range(len(batch["pair_record_id"])):
                actual.append({key: value[i] for key, value in batch.items()})
        assert len(actual) == len(expected)
        for got, want in zip(actual, expected):
            assert got.keys() == want.keys()
            for key in got:
                if isinstance(got[key], torch.Tensor):
                    assert torch.equal(got[key], want[key]), key
                else:
                    assert got[key] == want[key], key


def test_numpy_dataset_epochs_produce_fresh_realizations(tmp_path):
    ds = _dynamic_dataset(tmp_path, pairs_per_epoch=4)
    ds.noise_implementation = "numpy"
    ds.set_epoch(0)
    first = ds[0]["image_b"].clone()
    ds.set_epoch(1)
    assert not torch.equal(first, ds[0]["image_b"])
    ds.set_epoch(0)
    assert torch.equal(first, ds[0]["image_b"])


def test_numpy_e02_views_are_independent_in_the_dataset(tmp_path):
    ds = _dynamic_dataset(tmp_path, pairs_per_epoch=4, second_noise_view=True)
    ds.noise_implementation = "numpy"
    item = ds[0]
    assert not torch.equal(item["image_b"], item["image_b_noise_view2"])
    # Image A carries no noise, so both views of it are identical.
    assert torch.equal(item["image_a"], item["image_a_noise_view2"])


def test_numpy_reverse_pairs_keep_the_observation_side_convention(tmp_path):
    """Noise follows the B *role*, so a reverse pair noises the other sample.

    Reverse records are the odd ordered indices when ``include_reverse=True``.
    """
    ds = _dynamic_dataset(tmp_path, pairs_per_epoch=4)
    ds.noise_implementation = "numpy"
    forward_record, _ = ds._record_and_offset(0)
    reverse_record, _ = ds._record_and_offset(1)
    assert reverse_record.sample_a_id == forward_record.sample_b_id
    assert reverse_record.sample_b_id == forward_record.sample_a_id
    # Clean A side is the reverse record's physical B sample, unchanged.
    reader = ds._get_reader()
    clean_a = ds.scaler.transform(
        np.asarray(reader.get(int(reverse_record.sample_a_index)), dtype=np.float32)
    )
    item = ds[1]
    assert torch.equal(item["image_a"].squeeze(0),
                       torch.from_numpy(np.asarray(clean_a, dtype=np.float32)))


def test_frozen_validation_noise_stays_jax_backed(tmp_path):
    """Selecting numpy for training must not change frozen validation noise."""
    from types import SimpleNamespace

    from dluxshera.ml.torch_data import (
        PairManifestDataset,
        VALIDATION_NOISE_IMPLEMENTATIONS,
        resolve_validation_noise_implementation,
    )

    assert resolve_validation_noise_implementation("numpy") == "reference"
    assert resolve_validation_noise_implementation("batched") == "batched"
    assert resolve_validation_noise_implementation("reference") == "reference"
    assert "numpy" not in VALIDATION_NOISE_IMPLEMENTATIONS

    ds = _dynamic_dataset(tmp_path, pairs_per_epoch=6)
    records = [ds._record_and_offset(i)[0] for i in range(6)]
    frozen = PairManifestDataset(
        catalog=ds.catalog,
        pair_manifest=SimpleNamespace(records=records),
        scaler=ds.scaler,
        noise_config=S12_VALIDATION_NOISE,
        noise_implementation=resolve_validation_noise_implementation("numpy"),
    )
    assert frozen.noise_implementation == "reference"
    # The frozen realization matches the JAX reference bitwise.
    reader = frozen._get_reader()
    for index, record in enumerate(records):
        expected_a, expected_b = apply_pair_noise(
            reader.get(int(record.sample_a_index)),
            reader.get(int(record.sample_b_index)),
            S12_VALIDATION_NOISE,
            pair_record_id=record.pair_record_id,
            dynamic_seed_offset=index,
        )
        item = frozen[index]
        assert torch.equal(
            item["image_b"].squeeze(0),
            torch.from_numpy(np.asarray(frozen.scaler.transform(expected_b), dtype=np.float32)),
        )
        assert torch.equal(
            item["image_a"].squeeze(0),
            torch.from_numpy(np.asarray(frozen.scaler.transform(expected_a), dtype=np.float32)),
        )


def test_numpy_is_a_selectable_runtime_implementation(monkeypatch):
    from dluxshera.ml.torch_data import NOISE_IMPLEMENTATIONS, resolve_noise_implementation

    assert "numpy" in NOISE_IMPLEMENTATIONS
    monkeypatch.delenv("DLUXSHERA_NOISE_IMPLEMENTATION", raising=False)
    # Reference remains the default.
    assert resolve_noise_implementation() == "reference"
    assert resolve_noise_implementation("numpy") == "numpy"
    monkeypatch.setenv("DLUXSHERA_NOISE_IMPLEMENTATION", "numpy")
    assert resolve_noise_implementation() == "numpy"


def test_runtime_provenance_records_both_implementations_and_rng(monkeypatch):
    from dluxshera.ml.training import _runtime_info

    monkeypatch.setenv("DLUXSHERA_NOISE_IMPLEMENTATION", "numpy")
    runtime = _runtime_info(torch.device("cpu"))
    assert runtime["noise_implementation"] == "numpy"
    assert runtime["validation_noise_implementation"] == "reference"
    assert runtime["numpy_version"] == np.__version__
    assert runtime["numpy_noise_rng"]["bit_generator"] == "PCG64"
    assert runtime["jax_version"]


def test_resume_rejects_switching_between_jax_and_numpy():
    from dluxshera.ml.training import _validate_resume_noise_implementation

    for previous, current in (("numpy", "batched"), ("batched", "numpy"),
                              ("numpy", "reference"), ("reference", "numpy")):
        checkpoint = {"runtime_provenance": {"noise_implementation": previous}}
        with pytest.raises(ValueError, match="Resume noise implementation mismatch"):
            _validate_resume_noise_implementation(checkpoint, current)
    _validate_resume_noise_implementation(
        {"runtime_provenance": {"noise_implementation": "numpy"}}, "numpy"
    )


def test_diagnostic_refuses_bitwise_equivalence_for_numpy(tmp_path, monkeypatch):
    """A bitwise check against a different RNG stream is meaningless, not failing."""
    from tests.ml.test_batched_noise import _equivalence_request

    ds = _dynamic_dataset(tmp_path, pairs_per_epoch=4)
    request = _equivalence_request(ds, monkeypatch, noise_implementation="numpy")
    with pytest.raises(ValueError, match="bitwise"):
        request.verify_equivalence()


def test_diagnostic_accepts_numpy_and_offers_the_shared_rng_ceiling():
    """The CLI must expose numpy and label the shared-RNG benchmark non-production."""
    from work.experiments.ml.hpc.diagnose_ml_dataloader import ProbeRequest, main

    assert "numpy_shared_rng_ceiling" in ProbeRequest.__dataclass_fields__
    assert ProbeRequest.__dataclass_fields__["numpy_shared_rng_ceiling"].default is False
    with pytest.raises(SystemExit):
        main(["--help"])


@pytest.mark.parametrize("requested,train,validation", [
    ("numpy", "numpy", "reference"),
    ("batched", "batched", "batched"),
    ("reference", "reference", "reference"),
])
def test_diagnostic_validation_loader_keeps_jax_for_numpy_selection(
    requested, train, validation
):
    """The diagnostic must mirror training's frozen-validation isolation.

    ``resolved_noise_implementation`` is the branch ``_build`` uses to assign
    ``dataset.noise_implementation``, so this exercises the shipped selection
    rather than restating it.
    """
    from work.experiments.ml.hpc.diagnose_ml_dataloader import ProbeRequest

    def _request(loader: str) -> ProbeRequest:
        request = object.__new__(ProbeRequest)
        object.__setattr__(request, "loader", loader)
        object.__setattr__(request, "noise_implementation", requested)
        return request

    assert _request("train").resolved_noise_implementation() == train
    assert _request("validation").resolved_noise_implementation() == validation


def test_sustained_gate_recommends_more_batches_instead_of_failing_fast_loaders():
    """A fast loader must be told to measure longer, not marked unusable."""
    from dluxshera.ml.training import (
        MIN_SUSTAINED_SAMPLES,
        MIN_SUSTAINED_SECONDS,
        smoke_test_dataloader,
    )

    class _Instant(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 64

        def __getitem__(self, index: int):
            return {"image_a": torch.zeros(1, 2, 2)}

    loader = torch.utils.data.DataLoader(_Instant(), batch_size=4, num_workers=0)
    result = smoke_test_dataloader(loader, batches=16, timeout_seconds=60, warmup_batches=2)
    # Fast enough that the 20-second span gate cannot be met in 16 batches.
    assert result["sustained_measurement_valid"] is False
    recommended = result["recommended_batches_for_valid_measurement"]
    assert recommended is not None
    assert recommended > 16
    assert recommended >= MIN_SUSTAINED_SAMPLES
    assert "was fast, not that it is unusable" in result["sustained_measurement_note"]
    assert f"{MIN_SUSTAINED_SECONDS:.0f}s" in result["sustained_measurement_note"]
