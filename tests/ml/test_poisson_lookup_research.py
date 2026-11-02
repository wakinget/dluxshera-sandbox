"""Bounded checks for the Poisson-lookup research prototype.

These tests protect the *conclusions* of the lookup investigation, not a
production pathway.  The prototype is deliberately outside ``src/dluxshera``
and must never be imported by the S12 DataLoader; one test asserts that
separation directly.

Everything here uses small tables and modest draw counts so the suite stays
fast.  The statistical assertions are deliberately loose where sampling noise
dominates and tight only where the effect is structural (integer quantization),
because that is the finding that decides the question.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RESEARCH = Path(__file__).resolve().parents[2] / "work/experiments/ml/research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

poisson_lookup_prototype = pytest.importorskip("poisson_lookup_prototype")

DyadicPoissonTable = poisson_lookup_prototype.DyadicPoissonTable
PoissonTable = poisson_lookup_prototype.PoissonTable
dtype_is_safe_for_levels = poisson_lookup_prototype.dtype_is_safe_for_levels
integer_levels = poisson_lookup_prototype.integer_levels
level_coverage_report = poisson_lookup_prototype.level_coverage_report
sample_exact_numpy = poisson_lookup_prototype.sample_exact_numpy
sample_lookup = poisson_lookup_prototype.sample_lookup
table_bytes = poisson_lookup_prototype.table_bytes


def _small_table(levels: int = 64, side: int = 128, seed: int = 5) -> PoissonTable:
    return PoissonTable(levels=integer_levels(levels - 1), width=side,
                        height=side, dtype="uint16", seed=seed)


def test_prototype_is_not_reachable_from_production_noise():
    """The research module must stay out of the production import graph."""
    import dluxshera.ml.noise as noise
    import dluxshera.ml.torch_data as torch_data

    for module in (noise, torch_data):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "poisson_lookup" not in source
        assert "PoissonTable" not in source


def test_table_memory_accounting_is_exact():
    assert table_bytes(levels=1024, width=1024, height=1024, dtype="uint16") == 2 * 1024**3
    assert table_bytes(levels=1024, width=1024, height=1024, dtype="float32") == 4 * 1024**3
    table = _small_table(levels=32, side=64)
    assert table.nbytes == table_bytes(levels=32, width=64, height=64, dtype="uint16")
    assert table.describe()["draws_per_level"] == 64 * 64


def test_uint16_safety_margin_is_reported_not_assumed():
    """uint16 is safe at moderate rates and unsafe near its ceiling."""
    assert dtype_is_safe_for_levels(max_level=1023.0, dtype="uint16")["safe"] is True
    assert dtype_is_safe_for_levels(max_level=65000.0, dtype="uint16")["safe"] is False
    assert dtype_is_safe_for_levels(max_level=65000.0, dtype="float32")["integer"] is False


def test_table_generation_rejects_values_that_do_not_fit_the_dtype():
    with pytest.raises(ValueError, match="does not fit"):
        PoissonTable(levels=np.array([65000.0]), width=64, height=64,
                     dtype="uint16", seed=1)


@pytest.mark.parametrize("strategy", ["per_pixel", "offset"])
def test_lookup_preserves_shape_and_is_seed_reproducible(strategy):
    table = _small_table()
    expected = np.full((4, 16, 16), 20.0)
    first = sample_lookup(expected, table, strategy=strategy,
                          rng=np.random.default_rng(3))
    again = sample_lookup(expected, table, strategy=strategy,
                          rng=np.random.default_rng(3))
    assert first.shape == expected.shape
    assert first.dtype == np.float32
    assert np.array_equal(first, again)
    # A different seed must give a different realization, or the index
    # randomization is not doing anything.
    other = sample_lookup(expected, table, strategy=strategy,
                          rng=np.random.default_rng(4))
    assert not np.array_equal(first, other)


def test_lookup_accepts_a_bare_two_dimensional_image():
    """A single image has no leading batch axis; offset indexing must still work."""
    table = _small_table()
    got = sample_lookup(np.full((16, 16), 20.0), table, strategy="offset",
                        rng=np.random.default_rng(1))
    assert got.shape == (16, 16)


def test_lookup_rejects_unknown_strategies_and_policies():
    table = _small_table(levels=8, side=32)
    image = np.full((8, 8), 3.0)
    with pytest.raises(ValueError, match="strategy must be"):
        sample_lookup(image, table, strategy="nope", rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="level_policy must be"):
        sample_lookup(image, table, level_policy="nope",
                      rng=np.random.default_rng(0))


def test_nearest_level_lookup_is_structurally_wrong_below_one_count():
    """The decisive finding: integer levels destroy the faint-PSF regime.

    At ``lambda = 0.37`` -- an ordinary PSF-outskirt value for S12's 50 ms
    exposures -- nearest-integer lookup rounds to level 0 and returns identically
    zero.  The true zero probability is ``exp(-0.37) = 0.69``, so this is not a
    small bias; the distribution is gone. This is a property of the integer
    grid, not of sampling noise, so it is asserted exactly.
    """
    table = _small_table()
    got = sample_lookup(np.full((64, 64), 0.37), table, level_policy="nearest",
                        rng=np.random.default_rng(0))
    assert np.all(got == 0.0)
    assert float(np.mean(got == 0)) == 1.0
    assert abs(float(np.exp(-0.37)) - 0.69) < 0.01  # the value it should have had

    # lambda = 1.5 rounds up to 2, inflating the mean by a third.
    got = sample_lookup(np.full((128, 128), 1.5), table, level_policy="nearest",
                        rng=np.random.default_rng(0))
    assert got.mean() > 1.9


def test_mixture_policy_fixes_the_mean_but_overdisperses():
    """Mixture is unbiased in the mean and still not Poisson.

    A mixture of Poisson(k) and Poisson(k+1) with weight ``f`` has variance
    ``lambda + f(1-f)``, so at ``f = 0.5`` the variance is inflated by 0.25.
    At ``lambda = 0.5`` that is a 50% excess -- the Fano factor is no longer 1.
    """
    table = _small_table()
    expected = np.full((400, 400), 0.5)
    got = sample_lookup(expected, table, level_policy="mixture",
                        rng=np.random.default_rng(0))
    assert abs(got.mean() - 0.5) < 0.02
    # Theory: variance = 0.5 + 0.25 = 0.75, i.e. Fano = 1.5, not 1.
    assert got.var() > 0.65
    reference = sample_exact_numpy(expected, rng=np.random.default_rng(1))
    assert abs(reference.var() - 0.5) < 0.02
    assert got.var() > reference.var() * 1.2


def test_level_coverage_flags_clipped_bright_pixels():
    """Pixels above the top stored level get the wrong distribution entirely."""
    table = _small_table(levels=64)
    images = np.concatenate([
        np.full((1, 8, 8), 10.0),
        np.full((1, 8, 8), 5000.0),
    ])
    report = level_coverage_report(images, table)
    assert report["clipped_counts_are_wrong_distribution"] is True
    assert report["fraction_pixels_above_top_level"] == pytest.approx(0.5)
    assert report["levels_required_for_full_coverage"] == 5001
    # Covering a several-decade count range is what forces a large table.
    assert report["uint16_mib_for_full_coverage_at_this_spatial_size"] > 100.0

    within = level_coverage_report(np.full((2, 8, 8), 10.0), table)
    assert within["clipped_counts_are_wrong_distribution"] is False
    assert within["fraction_pixels_above_top_level"] == 0.0


def test_dyadic_superposition_is_distributionally_correct_on_lattice():
    """The one exact construction: moments match Poisson at lattice lambdas.

    Poisson is closed under addition of independent variates, so summing pool
    draws for each set dyadic bit is exactly Poisson at the quantized rate --
    unlike nearest/mixture, which are approximations by construction.
    """
    table = DyadicPoissonTable(delta=0.25, rates=10, width=512, height=512, seed=11)
    rng = np.random.default_rng(3)
    for lam in (0.25, 1.0, 5.0, 20.0):
        got, gathers = table.sample(np.full((500, 500), lam), rng=rng)
        assert gathers >= 1
        assert abs(got.mean() - lam) / lam < 0.02
        assert abs(got.var() - lam) / lam < 0.03
        # Zero probability is the moment-insensitive check that catches the
        # quantization failures above.
        assert abs(float(np.mean(got == 0)) - float(np.exp(-lam))) < 0.01


def test_duplicate_input_probe_exposes_the_shared_offset_defect():
    """Identical expected images must not receive the same "independent" draw.

    Correlation across *different* expected images cannot detect a shared
    index, because the expected values differ and so the outputs differ
    regardless.  Feeding a batch identical expected images isolates the
    indexing, and it shows that ``offset_per_level=True`` draws its
    translations once per count level and reuses them for every image in the
    call -- so two requested realizations of the same image are literally the
    same sample, not two independent ones.

    This is a limitation of the rejected prototype, recorded rather than fixed.
    """
    table = _small_table()
    duplicates = np.stack([np.full((16, 16), 20.0)] * 4)

    shared = sample_lookup(duplicates, table, strategy="offset",
                           offset_per_level=True, rng=np.random.default_rng(0))
    # Every pair is identical: one translation per level, reused batch-wide.
    assert all(
        np.array_equal(shared[i], shared[j])
        for i in range(4)
        for j in range(i + 1, 4)
    )

    # Per-image offsets and per-pixel indexing do not share this defect.
    for kwargs in (dict(strategy="offset", offset_per_level=False),
                   dict(strategy="per_pixel")):
        got = sample_lookup(duplicates, table, rng=np.random.default_rng(0), **kwargs)
        assert not any(
            np.array_equal(got[i], got[j])
            for i in range(4)
            for j in range(i + 1, 4)
        ), kwargs


def test_correlation_report_records_duplicate_input_independence():
    """The reported probe must agree with the behaviour above."""
    correlation_report = poisson_lookup_prototype.correlation_report
    table = _small_table()
    per_image = correlation_report(20.0, table, strategy="offset",
                                   level_policy="nearest", images=4,
                                   shape=(16, 16), seed=1)
    probe = per_image["duplicate_input_independence"]
    assert probe["independent"] is True
    assert probe["identical_output_pairs"] == []
    assert probe["images_in_probe"] == 4


def test_dyadic_exactness_is_conditional_not_unqualified():
    """The module must not describe the finite table as unconditionally exact.

    The dyadic construction is distributionally exact only at lattice lambdas
    and only under the idealization that summed pool entries are independent --
    which a finite table cannot supply across a campaign.
    """
    from pathlib import Path

    # Collapse whitespace so the check is insensitive to docstring line wrapping.
    source = " ".join(
        Path(poisson_lookup_prototype.__file__).read_text(encoding="utf-8").split()
    )
    assert "exact under the stated idealization" in source
    assert 'never as "equivalent to unlimited independent Poisson draws"' in source
    assert "A finite table cannot supply unlimited independent draws" in source
    assert "Shared-offset caveat" in source

    table = DyadicPoissonTable(delta=0.25, rates=8, width=64, height=64, seed=4)
    rng = np.random.default_rng(0)
    # Off-lattice lambda is quantized, not represented exactly.
    off_lattice = 0.3
    got, _ = table.sample(np.full((300, 300), off_lattice), rng=rng)
    assert abs(got.mean() - 0.25) < abs(got.mean() - off_lattice)
    # Above the representable maximum, lambda is clipped.
    clipped, _ = table.sample(np.full((200, 200), table.max_lambda * 4), rng=rng)
    assert clipped.mean() < table.max_lambda * 1.1


def test_dyadic_construction_costs_one_gather_per_set_bit():
    """Its cost, and therefore its viability, scales with the bit count."""
    table = DyadicPoissonTable(delta=1.0, rates=12, width=128, height=128, seed=2)
    rng = np.random.default_rng(0)
    _, one = table.sample(np.full((2, 8, 8), 1.0), rng=rng)
    _, many = table.sample(np.full((2, 8, 8), 4095.0), rng=rng)
    assert one == 1
    assert many == 12
    assert table.max_lambda == pytest.approx(4095.0)
